from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import datasets
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from chex import Array, Scalar
from jax.experimental import global_device_array as gda_lib
from jax.experimental.global_device_array import Device
from jax.experimental.maps import Mesh
from jax.sharding import Mesh, NamedSharding, PartitionSpec

P = PartitionSpec
Pytree = Any

data_dim = 0


def data_loader(
    dataset: datasets.Dataset,
    batch_size: Scalar,
    shape_dtypes: Pytree,
    *,
    rng: Optional[Array] = None,
    shuffle: bool = False,
    drop_last: bool = True,
):
    # shape_dtypes is a Pytree of jax.ShapeDtypeStruct
    if shuffle:
        assert rng is not None
        batch_idx = jrandom.permutation(rng, len(dataset))
        batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))
    if drop_last:
        batches_per_epoch = len(dataset) // batch_size
        batch_idx = batch_idx[: batches_per_epoch * batch_size]
        batch_idx = batch_idx.reshape((batches_per_epoch, batch_size))
    else:
        batches_per_epoch = int(np.ceil((len(dataset) / batch_size)))
        batch_idx = np.array_split(batch_idx, batches_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = jax.tree_util.tree_map(
            lambda shape_dtype, x: jnp.array(x, shape_dtype.dtype),
            shape_dtypes,
            batch,
        )

        yield batch


data_dim = 0


def check_inputs(dataset, global_data_shape, data_axes):
    # TODO(sholto): Is there a way to do this without calling dataset?
    dataset_structure = jax.tree_util.tree_structure(next(data_loader(dataset, 1)))
    global_data_shape_structure = jax.tree_util.tree_structure(global_data_shape)
    data_axes_structure = jax.tree_util.tree_structure(data_axes)

    try:
        assert (
            dataset_structure == global_data_shape_structure == data_axes_structure
        ), "All inputs should have the same pytree structure."
    except AssertionError as msg:
        (
            print(
                f"""{msg} - Dataset: {dataset_structure}, \n Shapes:
              {global_data_shape_structure}, \n Axes: {data_axes_structure}"""
            )
        )

    shapes, _ = jax.tree_util.tree_flatten(global_data_shape)
    batch_dims = [s.shape[0] for s in shapes]

    assert all(
        b == batch_dims[0] for b in batch_dims
    ), "All batch axis should be equal for gdas"

    assert all(
        b.shape[0] == shapes[0].shape[0] for b in shapes
    ), "All dataset elements should be sharded along the data axis identically"

    batch_dim = batch_dims[0]
    return batch_dim


def get_all_data_all_hosts_pipeline(
    dataset,
    global_data_shape: Pytree,
    global_mesh: Mesh,
    data_axes: Pytree,
) -> Callable[[], Pytree]:
    """Return the same, globally sized dataloader across all hosts."""

    batch_dim = check_inputs(dataset, global_data_shape, data_axes)

    device_to_index = jax.tree_map(
        lambda shape, axes: gda_lib.get_shard_indices(shape.shape, global_mesh, axes),
        global_data_shape,
        data_axes,
    )

    data = data_loader(dataset, batch_dim)

    next_fn = partial(
        get_next_all_data_all_hosts,
        data,
        device_to_index,
        global_data_shape,
        global_mesh,
        data_axes,
    )
    return next_fn


def get_next_all_data_all_hosts(
    dataset,
    device_to_index: Dict[Device, Tuple[slice, slice]],
    global_data_shape: Pytree,
    global_mesh: Mesh,
    data_axes: P,
) -> Pytree:
    """Fill device buffers with appropriate slice of the globally identical data."""
    batch = next(dataset)

    def form_global_array(element, shape, axes, device_to_index):
        device_buffers = [
            jax.device_put(element[device_to_index[device]], device)
            for device in jax.local_devices()
        ]
        sharding = NamedSharding(global_mesh, axes)
        global_array = jax.make_array_from_single_device_arrays(
            shape.shape,
            sharding,
            device_buffers,
        )
        return global_array

    global_arrays = jax.tree_map(
        form_global_array, batch, global_data_shape, data_axes, device_to_index
    )

    return global_arrays


@dataclass
class ShardInfo:
    idx: int
    size: int


def get_per_replica_data_pipeline(
    dataset: datasets.Dataset,
    global_data_shape: Pytree,
    global_mesh: Mesh,
    data_axes: Pytree,
) -> Callable[[], Array]:
    check_inputs(dataset, global_data_shape, data_axes)

    device_to_index = jax.tree_map(
        lambda shape, axes: gda_lib.get_shard_indices(shape.shape, global_mesh, axes),
        global_data_shape,
        data_axes,
    )

    shard_idx_to_dataset = {}

    def identify_shards(_, device_to_index) -> Dict[Device, ShardInfo]:
        index_hash_to_shard_idx: Dict[int, int] = {}
        device_to_shard_info: Dict[Device, int] = {}
        for (device, index_tuple) in device_to_index.items():
            index_hash = gda_lib._hashed_index(index_tuple)
            shard_idx = index_hash_to_shard_idx.setdefault(
                index_hash, len(index_hash_to_shard_idx)
            )
            indices_size = index_tuple[data_dim].stop - index_tuple[data_dim].start
            device_to_shard_info[device] = ShardInfo(shard_idx, indices_size)

        num_shards = len(index_hash_to_shard_idx)
        for device in jax.local_devices():
            shard_info = device_to_shard_info[device]
            if shard_info.idx not in shard_idx_to_dataset:
                shard_idx_to_dataset[shard_info.idx] = data_loader(
                    dataset.shard(num_shards=num_shards, index=shard_info.idx),
                    shard_info.size,
                )

        return device_to_shard_info

    device_to_shard_info = jax.tree_map(
        identify_shards, global_data_shape, device_to_index
    )

    next_fn = partial(
        get_next_per_replica,
        device_to_shard_info,
        shard_idx_to_dataset,
        global_data_shape,
        global_mesh,
        data_axes,
    )

    return next_fn


def transpose_and_wrap_per_shard(shard_idx_to_loaded_data: Dict[int, Pytree]) -> Pytree:
    outer_structure = jax.tree_util.tree_structure(
        {k: 0 for k in shard_idx_to_loaded_data}
    )
    inner_structure = jax.tree_util.tree_structure(
        next(iter(shard_idx_to_loaded_data.values()))
    )
    transposed_tree = jax.tree_util.tree_transpose(
        outer_structure, inner_structure, shard_idx_to_loaded_data
    )
    return transposed_tree


def get_next_per_replica(
    device_to_shard_info: Pytree,
    shard_idx_to_dataset: Dict[int, datasets.Dataset],
    global_data_shape: Pytree,
    global_mesh: Mesh,
    data_axes: Pytree,
) -> Array:
    """Gets the next batch of filled device_buffers using per replica pipelines."""
    shard_idx_to_loaded_data = {
        idx: next(dataset) for idx, dataset in shard_idx_to_dataset.items()
    }

    per_output_sharded_info = transpose_and_wrap_per_shard(shard_idx_to_loaded_data)

    def form_global_array(shape, output_sharded_info, device_to_shard_info, axes):
        device_buffers = []
        sharding = NamedSharding(global_mesh, axes)
        for idx, device in enumerate(jax.local_devices()):
            data_shard_info = device_to_shard_info[device]
            data = output_sharded_info[data_shard_info.idx]
            device_buffers.append(jax.device_put(data, device))
        global_array = jax.make_array_from_single_device_arrays(
            shape.shape,
            sharding,
            device_buffers,
        )
        return global_array

    global_arrays = jax.tree_map(
        form_global_array,
        global_data_shape,
        per_output_sharded_info,
        device_to_shard_info,
        data_axes,
    )

    return global_arrays


def get_unique_shards(
    host_to_devices: Dict[int, List[Device]],
    device_to_index: Dict[Device, Tuple[slice, slice]],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Looks at the sets of data each host needs, deduplicates, assigns a shard to the set."""

    host_to_dataset_shard = {}
    dataset_shard_hash_to_index = {}

    for host_id, host_devices in host_to_devices.items():
        host_indices = [device_to_index[device] for device in host_devices]
        hashable_indices = jax.tree_map(lambda s: (s.start, s.stop), host_indices)
        pipeline_hash = hash(tuple(set(hashable_indices)))
        # assign each host's set of indices a shard index in the order we discover
        # this will be the shard index loaded by data
        host_to_dataset_shard[host_id] = dataset_shard_hash_to_index.setdefault(
            pipeline_hash, len(dataset_shard_hash_to_index)
        )

    num_unique_shards = len(dataset_shard_hash_to_index)
    return host_to_dataset_shard, num_unique_shards


def convert_global_indices_to_local_indices(
    device_to_index: Dict[Device, Tuple[slice, slice]]
) -> Tuple[Dict[Device, slice], int]:
    """Converts global GDA indices for each device to local indices of host loaded data."""

    local_indices = [device_to_index[device] for device in jax.local_devices()]
    # Tacit assumption that we -only- shard dataset batch along data dim here, we could
    # relax this but I'm not sure it would actually be handled right by this approach:
    data_indices = [(s[data_dim].start, s[data_dim].stop) for s in local_indices]
    unique_slice_sizes = {idx: idx[1] - idx[0] for idx in data_indices}

    # assign a unique local data slice to each device
    total_data_to_load = 0
    device_index_hash_to_local_index = {}
    for idx, size in unique_slice_sizes.items():
        device_index_hash_to_local_index[idx] = slice(
            total_data_to_load, total_data_to_load + size
        )
        total_data_to_load += size

    device_to_local_indices = {}
    for device, data_index in zip(jax.local_devices(), data_indices):
        device_to_local_indices[device] = device_index_hash_to_local_index[data_index]

    return device_to_local_indices, total_data_to_load


def get_per_host_data_pipeline(
    dataset: datasets.Dataset,
    global_data_shape: np.ndarray,
    global_mesh: Mesh,
    data_axes: P,
) -> Callable[[], Pytree]:
    """One data pipeline per host.
    To do this, we determine which pieces of data each host needs to feed it's
    devices,
    identify the unique sets of these (which is likely < num_hosts), and then
    create
    a data pipeline for each set.
        + No overhead from multiple pipelines per host
        - High complexity
        - Doesn't allow for incomplete overlap in the batches loaded by hosts
    Args:
        dataset: dataset over all files
        global_data_shape: what the size of the GDA should be
        global_mesh: global deivces mesh
        data_axes: axes along which data is partitioned
        Returns:
        sharded_dataset: Correct dataset to load for this host
        host_local_indices: indices for just the data loaded by the host's pipeline
    """

    check_inputs(dataset, global_data_shape, data_axes)

    # pytree of 'device_to_index' objects matching the structure of data
    device_to_index = jax.tree_map(
        lambda shape, axes: gda_lib.get_shard_indices(shape.shape, global_mesh, axes),
        global_data_shape,
        data_axes,
    )

    # group by host_id
    host_to_devices = defaultdict(list)
    for d in jax.devices():
        host_to_devices[d.host_id].append(d)

    # Now, we want to find the number of unique (per host) dataset shards which
    # should be loaded and assign each host to their shard.

    # Now, as we are creating our own slice in this function, and assuming that
    # we only have one dimension we are sharding along, we don't need to do
    # clever tree mapping as the unique shards -> therefore just take
    # the first one and get the unique sharding from that.
    dataset_structure = jax.tree_util.tree_structure(global_data_shape)
    representative_device_to_index = dataset_structure.flatten_up_to(device_to_index)[0]
    host_to_dataset_shard, num_shards = get_unique_shards(
        host_to_devices, representative_device_to_index
    )
    # And assign devices indices into the data to be loaded by the host
    # The slices generated here are only along the batch dim, and thus will work
    # for all items in the data output pytree
    host_local_indices, total_data_to_load = convert_global_indices_to_local_indices(
        representative_device_to_index
    )

    # Create the data pipeline
    local_data_shard_index = host_to_dataset_shard[jax.process_index()]

    sharded_dataset = data_loader(
        dataset.shard(num_shards=num_shards, index=local_data_shard_index),
        total_data_to_load,
    )

    next_fn = partial(
        get_next_per_host,
        sharded_dataset,
        host_local_indices,
        global_data_shape,
        global_mesh,
        data_axes,
    )

    return next_fn


def get_next_per_host(
    sharded_dataset: datasets.Dataset,
    host_local_indices: Dict[Device, slice],
    global_data_shape: Pytree,
    global_mesh: Mesh,
    data_axes: P,
) -> Array:
    """Get device buffers to form GDA using per host pipeline."""

    # load from a single pipeline for the entire host
    # this is returned as a pytree in the same shape as global data shape
    local_data = next(sharded_dataset)
    # Slice this up using local indices and give it to the host local devices
    def form_global_array(element, shape, axes):
        device_buffers = []
        sharding = NamedSharding(global_mesh, axes)
        for idx, device in enumerate(jax.local_devices()):
            local_indices = host_local_indices[device]
            data = element[local_indices]
            device_buffers.append(jax.device_put(data, device))
        global_array = jax.make_array_from_single_device_arrays(
            shape.shape,
            sharding,
            device_buffers,
        )
        return global_array

    global_arrays = jax.tree_map(
        form_global_array, local_data, global_data_shape, data_axes
    )

    return global_arrays
