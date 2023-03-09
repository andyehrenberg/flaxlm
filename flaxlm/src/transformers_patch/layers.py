from typing import Callable, Optional, Tuple, Any, Union
import dataclasses

import jax.lax as lax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jax.random import PRNGKey
import flax.linen as nn
from flax.linen.dtypes import promote_dtype

import flaxlm.src.partitioning_utils as partitioning_utils


P = PartitionSpec
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

PrecisionLike = Union[
    None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]
]

default_kernel_init = nn.initializers.lecun_normal()


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
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.
        Args:
            inputs: The nd-array to be transformed.
        Returns:
            The transformed input.
        """
        kernel = self.param(
            'kernel',
            nn.with_logical_partitioning(self.kernel_init, self.names),
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )
        if self.use_bias:
            bias = self.param(
                'bias', self.bias_init, (self.features,), self.param_dtype)
        else:
            bias = None
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        # Parameter gather for FSDP
        mesh_axes = nn.logical_to_mesh_axes(self.names)
        if mesh_axes[0] == "data":
            kernel = partitioning_utils.with_logical_constraint(
                kernel, P(None, self.names[1])
            )
        elif mesh_axes[1] == "data":
            kernel = partitioning_utils.with_logical_constraint(
                kernel, P(self.names[0], None)
            )

        y = lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,),(0,)),((), ())),
            precision=self.precision,
        )
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y
    

default_embed_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)


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
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    embedding_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_embed_init
    embedding: Array = dataclasses.field(init=False)

    def setup(self):
        self.embedding = self.param(
            'embedding',
            nn.with_logical_partitioning(self.embedding_init, self.names),
            (self.num_embeddings, self.features),
            self.param_dtype
        )

    def __call__(self, inputs: Array) -> Array:
        """Embeds the inputs along the last dimension.
        Args:
            inputs: input data, all dimensions are considered batch dimensions.
        Returns:
            Output which is embedded input data.  The output shape follows the input,
            with an additional `features` dimension appended.
        """
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError('Input type must be an integer or unsigned integer.')
        # Use take because fancy indexing numpy arrays with JAX indices does not
        # work correctly.
        embedding, = promote_dtype(self.embedding, dtype=self.dtype, inexact=False)
        
        # FSDP gather
        mesh_axes = nn.logical_to_mesh_axes(self.names)
        if mesh_axes[0] == "data":
            embedding = partitioning_utils.with_logical_constraint(
                embedding, P(None, self.names[1])
            )
        elif mesh_axes[1] == "data":
            embedding = partitioning_utils.with_logical_constraint(
                embedding, P(self.names[0], None)
            )

        return jnp.take(embedding, inputs, axis=0)
