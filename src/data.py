from collections import defaultdict
from functools import cached_property, partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import datasets
import jax
import jax.experimental.multihost_utils as multihost_utils
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from chex import Array, Scalar
from jax.experimental import global_device_array as gda_lib
from jax.experimental.global_device_array import Device
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import flax.linen as nn

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


def pad_sequence(sequence, max_len, pad_value, pad_right):
    assert len(sequence) <= max_len, "sequence has size larger than max_len"

    pad_tokens = [pad_value for _ in range(max_len - len(sequence))]
    ones = [1 for _ in range(len(sequence))]
    zeros = [0 for _ in range(max_len - len(sequence))]

    if pad_right:
        return sequence + pad_tokens, ones + zeros
    else:
        return pad_tokens + sequence, zeros + ones


def preprocess_seq2seq(
    dataset: datasets.Dataset,
    tokenizer: Callable,
    num_workers: int,
    tokenize_batch_size: int,
    group_batch_size: int,
    input_ids_column_name: str,
    pad_value: int,
    pad_right: bool,
    max_len: int,
    trunc_end: bool,
    decoder_max_len: Optional[int] = None,
    decoder_trunc_end: bool = True,
    decoder_input_ids_column_name: Optional[str] = None,
    remove_columns: Optional[List[str]] = None,
):
    remove_columns = (
        remove_columns + [input_ids_column_name]
        if remove_columns
        else [input_ids_column_name]
    )

    if decoder_input_ids_column_name:
        remove_columns += [decoder_input_ids_column_name]

    def tokenize_function(examples):
        output = tokenizer(examples[input_ids_column_name])
        if decoder_input_ids_column_name:
            decoder_inputs = tokenizer(examples[decoder_input_ids_column_name])
            output["decoder_input_ids"] = decoder_inputs.pop("input_ids")
            output["decoder_attention_mask"] = decoder_inputs.pop("attention_mask")
        return output

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=tokenize_batch_size,
        num_proc=num_workers,
        remove_columns=remove_columns,
    )

    def block_sequences(examples):
        output = {"input_ids": [], "attention_mask": []}
        if decoder_input_ids_column_name:
            output = {**output, "decoder_input_ids": [], "decoder_attention_mask": []}
        for i in range(len(examples["input_ids"])):
            if trunc_end:
                new_tokens = examples["input_ids"][i][:max_len]
            else:
                new_tokens = examples["input_ids"][i][-max_len:]
            padded, mask = pad_sequence(new_tokens, max_len, pad_value, pad_right)
            output["input_ids"].append(padded)
            output["attention_mask"].append(mask)
            if decoder_input_ids_column_name:
                if decoder_trunc_end:
                    new_tokens = examples["decoder_input_ids"][i][:decoder_max_len]
                else:
                    new_tokens = examples["decoder_input_ids"][i][-decoder_max_len:]
                padded, mask = pad_sequence(
                    new_tokens, decoder_max_len, pad_value, pad_right
                )
                output["decoder_input_ids"].append(padded)
                output["decoder_attention_mask"].append(mask)

        return output

    data = tokenized_dataset.map(
        block_sequences,
        batched=True,
        batch_size=group_batch_size,
        num_proc=num_workers,
    )

    return data


def preprocess_clm(
    dataset: datasets.Dataset,
    tokenizer: Callable,
    num_workers: int,
    block_size: int,
    tokenize_batch_size: int,
    group_batch_size: int,
    input_ids_column_name: str,
    decoder_input_ids_column_name: Optional[str] = None,
    remove_columns: Optional[List[str]] = None,
):
    remove_columns = (
        remove_columns + [input_ids_column_name]
        if remove_columns
        else [input_ids_column_name]
    )

    if decoder_input_ids_column_name:
        remove_columns += [decoder_input_ids_column_name]

    def tokenize_function(examples):
        output = tokenizer(examples[input_ids_column_name])
        if decoder_input_ids_column_name:
            decoder_inputs = tokenizer(examples[decoder_input_ids_column_name])
            output["decoder_input_ids"] = decoder_inputs.pop("input_ids")
            output["decoder_attention_mask"] = decoder_inputs.pop("attention_mask")
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
    host_batch_size = 0
    device_index_hash_to_local_index = {}
    for idx, size in unique_slice_sizes.items():
        device_index_hash_to_local_index[idx] = slice(
            host_batch_size, host_batch_size + size
        )
        host_batch_size += size

    device_to_local_indices = {}
    for device, data_index in zip(jax.local_devices(), data_indices):
        device_to_local_indices[device] = device_index_hash_to_local_index[data_index]

    return device_to_local_indices, host_batch_size


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

    mesh_data_axes = nn.logical_to_mesh_axes(data_axes)

    global_arrays = jax.tree_map(
        form_global_array, local_data, global_data_shape, mesh_data_axes
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
        input_ids_column_name: str,
        remove_columns: str,
        num_workers: int,
        tokenize_batch_size: int,
        group_batch_size: int,
        mode: str = "seq2seq",
        dataset_name: Optional[str] = None,
        dataset_split: Optional[str] = None,
        decoder_input_ids_column_name: Optional[str] = None,
        block_size: Optional[int] = None,
        pad_value: Optional[int] = None,
        pad_right: Optional[bool] = None,
        max_len: Optional[int] = None,
        trunc_end: Optional[bool] = None,
        decoder_max_len: Optional[int] = None,
        decoder_trunc_end: bool = True,
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
            self.host_batch_size,
        ) = convert_global_indices_to_local_indices(representative_device_to_index)

        print("Host local indices: ", self.host_local_indices)
        print("Host batch size: ", self.host_batch_size)

        local_data_shard_index = host_to_dataset_shard[jax.process_index()]

        print("Local data shard index: ", local_data_shard_index)

        if isinstance(dataset, str):
            dataset = datasets.load_dataset(dataset, dataset_name, split=dataset_split)

        if mode == "seq2seq":
            preprocess_fn = partial(
                preprocess_seq2seq,
                pad_value=pad_value,
                pad_right=pad_right,
                max_len=max_len,
                trunc_end=trunc_end,
                decoder_max_len=decoder_max_len,
                decoder_trunc_end=decoder_trunc_end,
            )
        elif mode == "clm":
            preprocess_fn = partial(
                preprocess_clm,
                block_size=block_size,
            )
        else:
            raise NotImplementedError("Mode can either be seq2seq or clm")

        self.sharded_dataset = preprocess_fn(
            dataset.shard(num_shards=num_shards, index=local_data_shard_index),
            tokenizer=tokenizer,
            input_ids_column_name=input_ids_column_name,
            remove_columns=remove_columns,
            num_workers=num_workers,
            tokenize_batch_size=tokenize_batch_size,
            group_batch_size=group_batch_size,
            decoder_input_ids_column_name=decoder_input_ids_column_name,
        )

        check_inputs(self.sharded_dataset, self.global_data_shape, self.data_axes)

    def set_epoch(self, rng):
        loader = data_loader(
            self.sharded_dataset,
            self.host_batch_size,
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
        local_length = len(self.sharded_dataset) // self.host_batch_size
        all_lengths = multihost_utils.process_allgather(jnp.array(local_length))
        return int(jnp.min(all_lengths))
