from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding

import flax.linen as nn


def shard_logically_partitioned_params(params, mesh):
    # Given params with LogicallyPartitioned axis metadata, partition them
    return jax.tree_util.tree_map(
        lambda x: nn.LogicallyPartitioned(
            value=jax.device_put(
                x.value, NamedSharding(mesh, nn.logical_to_mesh_axes(x.names))
            ),
            names=x.names,
        ),
        params,
        is_leaf=lambda x: isinstance(x, nn.LogicallyPartitioned),
    )


def constraint_tree(tree, tree_spec):
    tree = jax.tree_util.tree_map()


def make_param_partitionable(param, mp_num, dim):
    if mp_num == 1:
        return param
    shape = param.shape
    original_length = param.shape[dim]
    remainder = original_length % mp_num
    new_length = original_length + mp_num - remainder
    shape = shape[:dim] + (new_length,) + shape[dim + 1 :]

    new_param = jnp.zeros(shape)
    new_param = jax.lax.dynamic_update_slice(
        new_param,
        param,
        (0,) * len(shape),
    )

    return new_param


def make_partitioning_rules(
    activation_partitioning_dims,
    parameter_partitioning_dims,
):
    """Gives default sharding rules in terms of logical axis names.

    Args:
    - activation_partitioning_dims: enables 2-D activation sharding when set to 2.
    - parameter_partitioning_dims: enables 2-D parameter sharding when set to 2.
    Returns:
    - Sequence of logical axis rules (`logical_name`, `shard_axis`) where dimensions of tensors annotated with `logical_name` will be sharded along `shard_axis`.
    """
    if activation_partitioning_dims == 0 and parameter_partitioning_dims == 0:
        rules = (
            ("batch", None),
            ("vocab", None),
            ("embed", None),
            ("mlp", None),
            ("heads", None),
            ("kv", None),
            ("joined_kv", None),
        )
    elif activation_partitioning_dims == 1 and parameter_partitioning_dims == 0:
        rules = (
            ("batch", "data"),
            ("vocab", None),
            ("embed", None),
            ("mlp", None),
            ("heads", None),
            ("kv", None),
            ("joined_kv", None),
        )
    elif activation_partitioning_dims == 1 and parameter_partitioning_dims == 1:
        rules = (
            ("batch", "data"),
            ("vocab", "model"),
            ("embed", None),
            ("mlp", "model"),
            ("heads", None),
            ("kv", None),
            ("joined_kv", "model"),
        )
    elif activation_partitioning_dims == 2 and parameter_partitioning_dims == 1:
        rules = (
            ("batch", "data"),
            ("vocab", "model"),
            ("mlp", "model"),
            ("heads", "model"),
            ("kv", None),
            ("joined_kv", "model"),
            ("embed", "model"),
        )
    elif activation_partitioning_dims == 1 and parameter_partitioning_dims == 2:
        rules = (
            ("batch", "data"),
            ("vocab", "model"),
            ("mlp", "model"),
            ("heads", "model"),
            ("kv", None),
            ("joined_kv", "model"),
            ("embed", "data"),
        )
    elif activation_partitioning_dims == 2 and parameter_partitioning_dims == 2:
        rules = (
            ("batch", "data"),
            ("vocab", "model"),
            ("mlp", "model"),
            ("heads", "model"),
            ("kv", None),
            ("joined_kv", "model"),
            ("embed", "model"),
            ("embed", "data"),
        )

    replicated_rules = (("length", None),)

    rules += replicated_rules

    return rules


# inspired by https://github.com/borisdayma/dalle-mini/blob/main/tools/train/train.py
def convert_per_device_batch_size(per_device_batch_size, mp_num, grad_accum_steps=1):
    per_node_per_grad_step_batch_size = (
        per_device_batch_size * jax.local_device_count() // mp_num
    )
    per_node_batch_size = per_node_per_grad_step_batch_size * grad_accum_steps

    batch_size = per_node_batch_size * jax.process_count()

    return batch_size


# taken from https://github.com/stanford-crfm/levanter/blob/main/src/levanter/mesh.py
def local_device_grid_positions(
    mesh, process_index: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a tuple of nd arrays, one for each axis, indicating the position of each device on the grid.
    Analogous to what np.where would return."""
    pi = process_index or jax.process_index()
    my_device_pos = np.vectorize(lambda dev: dev.process_index == pi)(mesh.devices)
    return my_device_pos.nonzero()


# taken from https://github.com/stanford-crfm/levanter/blob/main/src/levanter/mesh.py
def process_mesh_position(mesh, process_index: Optional[int] = None) -> Tuple[int, int]:
    """
    If we envision each process as a subgrid of the mesh for its devices, this is the position of the process
    in the coarsened process-level mesh
    """
    upper_left_position = np.array(
        [np.min(axis) for axis in local_device_grid_positions(mesh, process_index)]
    )
    local_mesh_size = mesh.local_mesh.devices.shape
    pos = upper_left_position // local_mesh_size
    # TODO: this assumes 2D mesh and contiguous devices per process
    assert len(pos) == 2
    return pos[0], pos[1]


# taken from https://github.com/stanford-crfm/levanter/blob/main/src/levanter/mesh.py
def process_mesh_size(mesh: Mesh) -> Tuple[int, int]:
    """
    If we envision each process as a subgrid of the mesh for its devices, then there is a process grid that
    is a coarsened version of the mesh. This is the size of the process grid.
    """
    local_mesh_size = mesh.local_mesh.devices.shape
    assert len(local_mesh_size) == 2
    assert mesh.devices.shape[0] % local_mesh_size[0] == 0
    assert mesh.devices.shape[1] % local_mesh_size[1] == 0
    return (
        mesh.devices.shape[0] // local_mesh_size[0],
        mesh.devices.shape[1] // local_mesh_size[1],
    )


# inspired by https://github.com/Sea-Snell/JAXSeq/blob/main/src/shard.py
def shard_data_list(data: List[Any], mesh: Mesh, dp_axis: int):
    dp_size = process_mesh_size(mesh)[dp_axis]
    dp_idx = process_mesh_position(mesh, jax.process_index())[dp_axis]
    return data[dp_idx::dp_size]


def convert_global_batch_size(bsize: int, mesh: Mesh, dp_axis: int, mp_num: int) -> int:
    dp_size = process_mesh_size(mesh)[dp_axis]
    assert (
        bsize % dp_size
    ) == 0, "batch size must be divisible by the number of data parallel hosts"
    loader_batch_size = bsize // dp_size
    per_device_batch_size = loader_batch_size // max(
        1, jax.local_device_count() // mp_num
    )
    return loader_batch_size, per_device_batch_size
