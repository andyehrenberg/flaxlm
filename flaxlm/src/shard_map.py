from functools import partial
from typing import Any, Callable, Optional, Tuple
import dataclasses

import flax.linen as nn
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
import jax
from jax import lax
import jax.experimental.shard_map as shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec


P = PartitionSpec

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

default_kernel_init = initializers.lecun_normal()
default_embed_init = initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)


def dot_gather_0(x, y):
    gathered = lax.all_gather(x, "data", axis=0, tiled=True)

    return jnp.dot(y, gathered)


def dot_gather_1(x, y):
    gathered = lax.all_gather(x, "data", axis=1, tiled=True)

    return jnp.dot(y, gathered)


def dot_gather_accum_0(kernel, inputs):
    axis_size = lax.psum(1, axis_name="data")
    axis_index = lax.axis_index(axis_name="data")
    chunk_size = kernel.shape[0]

    def f(i, carrys):
        accum, kernel = carrys
        x = lax.dynamic_slice(
            inputs,
            (0, 0, ((axis_index + i) % axis_size) * chunk_size),
            (inputs.shape[0], inputs.shape[1], chunk_size),
        )

        update = jnp.dot(x, kernel)

        kernel = lax.ppermute(
            kernel,
            axis_name="data",
            perm=[(j, (j - 1) % axis_size) for j in range(axis_size)],
        )

        accum = accum + update

        return accum, kernel

    accum = jnp.zeros(
        (inputs.shape[0], inputs.shape[1], kernel.shape[1]), dtype=kernel.dtype
    )
    # accum, kernel = jax.lax.fori_loop(0, axis_size - 1, f, (accum, kernel))
    for i in range(0, axis_size - 1):
        accum, kernel = f(i, (accum, kernel))

    i = axis_size - 1
    x = lax.dynamic_slice(
        inputs,
        (0, 0, ((axis_index + i) % axis_size) * chunk_size),
        (inputs.shape[0], inputs.shape[1], chunk_size),
    )
    update = jnp.dot(x, kernel)

    accum = accum + update

    return accum


def dot_gather_accum_1(kernel, inputs):
    axis_size = lax.psum(1, axis_name="data")
    axis_index = lax.axis_index(axis_name="data")
    chunk_size = kernel.shape[1]

    def f(i, carrys):
        accum, kernel = carrys

        update = jnp.dot(inputs, kernel)

        kernel = lax.ppermute(
            kernel,
            axis_name="data",
            perm=[(j, (j - 1) % axis_size) for j in range(axis_size)],
        )

        update_index = (0, 0, ((axis_index + i) % axis_size) * chunk_size)
        accum = lax.dynamic_update_slice(accum, update, update_index)

        return accum, kernel

    accum = jnp.zeros(
        (inputs.shape[0], inputs.shape[1], chunk_size * axis_size), dtype=kernel.dtype
    )
    # accum, kernel = jax.lax.fori_loop(0, axis_size - 1, f, (accum, kernel))
    for i in range(0, axis_size - 1):
        accum, kernel = f(i, (accum, kernel))

    update = jnp.dot(inputs, kernel)

    i = axis_size - 1
    update_index = (0, 0, ((axis_index + i) % axis_size) * chunk_size)
    accum = lax.dynamic_update_slice(accum, update, update_index)

    return accum


def bias_gather(x, y):
    x = lax.all_gather(x, "data", axis=0, tiled=True)
    x = jnp.reshape(x, (1,) * (y.ndim - 1) + (-1,))
    return x + y


def bias_gather_accum(bias, inputs):
    axis_size = lax.psum(1, axis_name="data")
    axis_index = lax.axis_index(axis_name="data")

    bias = jnp.reshape(bias, (1,) * (inputs.ndim - 1) + (-1,))
    chunk_size = bias.shape[-1]

    def f(i, carrys):
        accum, bias = carrys

        x = lax.dynamic_slice(
            inputs,
            (0, 0, ((axis_index + i) % axis_size) * chunk_size),
            (inputs.shape[0], inputs.shape[1], chunk_size),
        )

        update = x + bias

        bias = lax.ppermute(
            bias,
            axis_name="data",
            perm=[(j, (j - 1) % axis_size) for j in range(axis_size)],
        )

        update_index = (0, 0, ((axis_index + i) % axis_size) * chunk_size)
        accum = lax.dynamic_update_slice(accum, update, update_index)

        return accum, bias

    accum = jnp.zeros_like(inputs, dtype=bias.dtype)
    # accum, bias = jax.lax.fori_loop(0, axis_size - 1, f, (accum, bias))
    for i in range(0, axis_size - 1):
        accum, bias = f(i, (accum, bias))

    i = axis_size - 1
    x = lax.dynamic_slice(
        inputs,
        (0, 0, ((axis_index + i) % axis_size) * chunk_size),
        (inputs.shape[0], inputs.shape[1], chunk_size),
    )
    update = x + bias

    update_index = (0, 0, ((axis_index + i) % axis_size) * chunk_size)
    accum = lax.dynamic_update_slice(accum, update, update_index)

    return accum


class Dense(nn.Module):
    """A linear transformation applied over the last dimension of the input.
    Attributes:
      features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
    """

    features: int
    names: Tuple[str]
    use_bias: bool = True
    mesh: Optional[Mesh] = None
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.
        Args:
          inputs: The nd-array to be transformed.
        Returns:
          The transformed input.
        """
        kernel = self.param(
            "kernel",
            nn.with_logical_partitioning(self.kernel_init, self.names),
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )
        if self.use_bias:
            bias = self.param(
                "bias",
                nn.with_logical_partitioning(self.bias_init, self.names[-1]),
                (self.features,),
                self.param_dtype,
            )
        else:
            bias = None
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        mesh_axes = nn.logical_to_mesh_axes(self.names)
        if mesh_axes[0] == "data":
            dot_fn = shard_map.shard_map(
                dot_gather_accum_0,
                in_specs=(mesh_axes, P("data")),
                out_specs=P("data"),
                mesh=self.mesh,
                check_rep=False,
            )
        elif mesh_axes[1] == "data":
            dot_fn = shard_map.shard_map(
                dot_gather_accum_1,
                in_specs=(mesh_axes, P("data")),
                out_specs=P("data"),
                mesh=self.mesh,
                check_rep=False,
            )
        else:
            dot_fn = lambda x, y: jnp.dot(y, x)

        y = dot_fn(kernel, inputs)

        if bias is not None:
            if mesh_axes[1] == "data":
                y = shard_map.shard_map(
                    bias_gather_accum,
                    in_specs=(P("data"), P("data")),
                    out_specs=P("data"),
                    mesh=self.mesh,
                )(bias, y)
            else:
                y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


def take_gather_0(embedding, inputs):
    gathered = lax.all_gather(
        embedding,
        "data",
        axis=0,
        tiled=True,
    )
    inputs = jax.nn.one_hot(inputs, gathered.shape[0])

    return jnp.dot(inputs, gathered)


def take_gather_accum_0(kernel, inputs):
    axis_size = lax.psum(1, axis_name="data")
    axis_index = lax.axis_index(axis_name="data")
    chunk_size = kernel.shape[0]

    inputs = jax.nn.one_hot(inputs, chunk_size * axis_size)
    (inputs,) = promote_dtype(inputs, dtype=kernel.dtype, inexact=False)

    def f(i, carrys):
        accum, kernel = carrys
        x = lax.dynamic_slice(
            inputs,
            (0, 0, ((axis_index + i) % axis_size) * chunk_size),
            (inputs.shape[0], inputs.shape[1], chunk_size),
        )

        update = jnp.dot(x, kernel)

        kernel = lax.ppermute(
            kernel,
            axis_name="data",
            perm=[(j, (j - 1) % axis_size) for j in range(axis_size)],
        )

        accum = accum + update

        return accum, kernel

    accum = jnp.zeros(
        (inputs.shape[0], inputs.shape[1], kernel.shape[1]), dtype=kernel.dtype
    )
    # accum, kernel = jax.lax.fori_loop(0, axis_size - 1, f, (accum, kernel))
    for i in range(0, axis_size - 1):
        accum, kernel = f(i, (accum, kernel))

    i = axis_size - 1
    x = lax.dynamic_slice(
        inputs,
        (0, 0, ((axis_index + i) % axis_size) * chunk_size),
        (inputs.shape[0], inputs.shape[1], chunk_size),
    )
    update = jnp.dot(x, kernel)

    accum = accum + update

    return accum


def take_gather_1(embedding, inputs):
    gathered = lax.all_gather(
        embedding,
        "data",
        axis=1,
        tiled=True,
    )
    inputs = jax.nn.one_hot(inputs, gathered.shape[0])

    return jnp.dot(inputs, gathered)


def take_gather_accum_1(kernel, inputs):
    axis_size = lax.psum(1, axis_name="data")
    axis_index = lax.axis_index(axis_name="data")
    chunk_size = kernel.shape[1]

    inputs = jax.nn.one_hot(inputs, kernel.shape[0])
    (inputs,) = promote_dtype(inputs, dtype=kernel.dtype, inexact=False)

    def f(i, carrys):
        accum, kernel = carrys

        update = jnp.dot(inputs, kernel)

        kernel = lax.ppermute(
            kernel,
            axis_name="data",
            perm=[(j, (j - 1) % axis_size) for j in range(axis_size)],
        )

        update_index = (0, 0, ((axis_index + i) % axis_size) * chunk_size)
        accum = lax.dynamic_update_slice(accum, update, update_index)

        return accum, kernel

    accum = jnp.zeros(
        (inputs.shape[0], inputs.shape[1], chunk_size * axis_size), dtype=kernel.dtype
    )
    # accum, kernel = jax.lax.fori_loop(0, axis_size - 1, f, (accum, kernel))
    for i in range(0, axis_size - 1):
        accum, kernel = f(i, (accum, kernel))

    update = jnp.dot(inputs, kernel)

    i = axis_size - 1
    update_index = (0, 0, ((axis_index + i) % axis_size) * chunk_size)
    accum = lax.dynamic_update_slice(accum, update, update_index)

    return accum


class Embed(nn.Module):
    """Embedding Module.

    A parameterized function from integers [0, n) to d-dimensional vectors.

    Attributes:
        num_embeddings: number of embeddings.
        features: number of feature dimensions for each embedding.
        dtype: the dtype of the embedding vectors (default: same as embedding).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        embedding_init: embedding initializer.
    """

    num_embeddings: int
    features: int
    names: Tuple[str]
    mesh: Optional[Mesh] = None
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    embedding_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_embed_init
    embedding: Array = dataclasses.field(init=False)

    def setup(self):
        self.embedding = self.param(
            "embedding",
            nn.with_logical_partitioning(self.embedding_init, self.names),
            (self.num_embeddings, self.features),
            self.param_dtype,
        )
        mesh_axes = nn.logical_to_mesh_axes(self.names)
        if mesh_axes[0] == "data":
            self.embed_fn = shard_map.shard_map(
                take_gather_accum_0,
                in_specs=(mesh_axes, P("data")),
                out_specs=P("data"),
                mesh=self.mesh,
                check_rep=False,
            )
        elif mesh_axes[1] == "data":
            self.embed_fn = shard_map.shard_map(
                take_gather_accum_1,
                in_specs=(mesh_axes, P("data")),
                out_specs=P("data"),
                mesh=self.mesh,
                check_rep=False,
            )
        else:
            self.embed_fn = partial(jnp.take, axis=0)

    def __call__(self, inputs: Array) -> Array:
        """Embeds the inputs along the last dimension.

        Args:
        inputs: input data, all dimensions are considered batch dimensions.

        Returns:
        Output which is embedded input data.  The output shape follows the input,
        with an additional `features` dimension appended.
        """
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        # Use take because fancy indexing numpy arrays with JAX indices does not
        # work correctly.
        (embedding,) = promote_dtype(self.embedding, dtype=self.dtype, inexact=False)
        return self.embed_fn(embedding, inputs)
