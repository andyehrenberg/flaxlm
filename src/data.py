from collections import defaultdict
from functools import partial, cached_property
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import datasets
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from chex import Array, Scalar
from jax.experimental import global_device_array as gda_lib
from jax.experimental.global_device_array import Device
from jax.experimental.maps import Mesh
import jax.experimental.multihost_utils as multihost_utils
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
    max_steps: int = None,
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

    if max_steps:
        batch_idx = batch_idx[:max_steps]

    for idx in batch_idx:
        batch = dataset[idx]
        batch = jax.tree_util.tree_map(
            lambda shape_dtype, x: jnp.array(x, shape_dtype.dtype),
            shape_dtypes,
            batch,
        )
        # if "attention_mask" in batch.keys():
        #    batch["position_ids"] = batch["attention_mask"].cumsum(-1) - 1
        #    batch["position_ids"] = jnp.where(
        #        batch["attention_mask"] > 0, batch["position_ids"], 0
        #    )
        # if "decoder_attention_mask" in batch.keys():
        #    batch["decoder_position_ids"] = (
        #        batch["decoder_attention_mask"].cumsum(-1) - 1
        #    )
        #    batch["decoder_position_ids"] = jnp.where(
        #        batch["decoder_attention_mask"] > 0, batch["decoder_position_ids"], 0
        #    )

        yield batch


def preprocess(
    dataset: datasets.Dataset,
    tokenizer: Callable,
    text_column_name: str,
    remove_columns: str,
    num_workers: int,
    block_size: int,
    tokenize_batch_size: int,
    group_batch_size: int,
):
    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        return output

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=tokenize_batch_size,
        num_proc=num_workers,
        remove_columns=remove_columns,
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    data = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=group_batch_size,
        num_proc=num_workers,
    )

    return data


def check_inputs(dataset, global_data_shape, data_axes):
    dataset_structure = jax.tree_util.tree_structure(
        next(data_loader(dataset, 1, global_data_shape))
    )
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


def get_next_per_host(
    dataloader: Iterator,
    host_local_indices: Dict[Device, slice],
    global_data_shape: Pytree,
    global_mesh: Mesh,
    data_axes: P,
) -> Array:
    local_data = next(dataloader)

    def form_global_array(element, shape, axes):
        device_buffers = []
        sharding = NamedSharding(global_mesh, axes)
        for device in jax.local_devices():
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


class PerHostDataset:
    def __init__(
        self,
        dataset: Union[datasets.Dataset, str],
        global_data_shape: Pytree,
        global_mesh: Mesh,
        data_axes: P,
        tokenizer: Callable,
        text_column_name: str,
        remove_columns: str,
        num_workers: int,
        block_size: int,
        tokenize_batch_size: int,
        group_batch_size: int,
    ):
        self.global_data_shape = global_data_shape
        self.global_mesh = global_mesh
        self.data_axes = data_axes

        device_to_index = jax.tree_map(
            lambda shape, axes: gda_lib.get_shard_indices(
                shape.shape, global_mesh, axes
            ),
            self.global_data_shape,
            self.data_axes,
        )

        # group by host_id
        host_to_devices = defaultdict(list)
        for d in jax.devices():
            host_to_devices[d.host_id].append(d)

        # Assign host to dataset shard
        dataset_structure = jax.tree_util.tree_structure(global_data_shape)
        representative_device_to_index = dataset_structure.flatten_up_to(
            device_to_index
        )[0]
        host_to_dataset_shard, num_shards = get_unique_shards(
            host_to_devices, representative_device_to_index
        )
        print("Host to dataset: ", host_to_dataset_shard)
        (
            self.host_local_indices,
            self.total_data_to_load,
        ) = convert_global_indices_to_local_indices(representative_device_to_index)

        print("Host local indices: ", self.host_local_indices)
        print("Host batch size: ", self.total_data_to_load)

        local_data_shard_index = host_to_dataset_shard[jax.process_index()]

        print("Local data shard index: ", local_data_shard_index)

        if isinstance(dataset, str):
            dataset = datasets.load_dataset(dataset)

        self.sharded_dataset = preprocess(
            dataset.shard(num_shards=num_shards, index=local_data_shard_index),
            tokenizer,
            text_column_name,
            remove_columns,
            num_workers,
            block_size,
            tokenize_batch_size,
            group_batch_size,
        )

        check_inputs(self.sharded_dataset, self.global_data_shape, self.data_axes)

    def set_epoch(self, rng):
        loader = data_loader(
            self.sharded_dataset,
            self.total_data_to_load,
            self.global_data_shape,
            rng=rng,
            shuffle=True,
            max_steps=self._global_min_length,
        )

        next_fn = partial(
            get_next_per_host,
            loader,
            self.host_local_indices,
            self.global_data_shape,
            self.global_mesh,
            self.data_axes,
        )

        for step in range(self._global_min_length):
            yield next_fn()

    @cached_property
    def _global_min_length(self):
        local_length = len(self.sharded_dataset) // self.total_data_to_load
        all_lengths = multihost_utils.process_allgather(jnp.array(local_length))
        return int(jnp.min(all_lengths))
