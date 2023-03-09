from typing import Callable, Optional, Iterable, Tuple, Any, Union
import dataclasses

import jax.lax as lax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jax.random import PRNGKey
import flax.linen as nn
from flax.linen.dtypes import promote_dtype, canonicalize_dtype

import flaxlm.src.partitioning_utils as partitioning_utils


P = PartitionSpec
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
Axes = Union[int, Any]

PrecisionLike = Union[
    None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]
]

default_kernel_init = nn.initializers.lecun_normal()


def _canonicalize_axes(rank: int, axes: Axes) -> Tuple[int, ...]:
    """Returns a tuple of deduplicated, sorted, and positive axes."""
    if not isinstance(axes, Iterable):
        axes = (axes,)
    return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))


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
                'bias',
                nn.with_logical_partitioning(self.bias_init, (self.names[1],)),
                (self.features,),
                self.param_dtype,
            )
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
            if mesh_axes[1] == "data":
                bias = partitioning_utils.with_logical_constraint(
                    bias, P(self.names[1],)
                )
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

def _normalize(
    mdl: nn.Module,
    x: Array,
    mean: Array,
    var: Array,
    reduction_axes: Axes,
    feature_axes: Axes,
    dtype: Dtype,
    param_dtype: Dtype,
    epsilon: float,
    use_bias: bool, use_scale: bool,
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array],
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array],
):
    """"Normalizes the input of a normalization layer and optionally applies a learned scale and bias.
    Arguments:
        mdl: Module to apply the normalization in (normalization params will reside
        in this module).
        x: The input.
        mean: Mean to use for normalization.
        var: Variance to use for normalization.
        reduction_axes: The axes in ``x`` to reduce.
        feature_axes: Axes containing features. A separate bias and scale is learned
        for each specified feature.
        dtype: The dtype of the result (default: infer from input and params).
        param_dtype: The dtype of the parameters.
        epsilon: Normalization epsilon.
        use_bias: If true, add a bias term to the output.
        use_scale: If true, scale the output.
        bias_init: Initialization function for the bias term.
        scale_init: Initialization function for the scaling function.
    Returns:
        The normalized input.
    """
    reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
    feature_axes = _canonicalize_axes(x.ndim, feature_axes)
    stats_shape = list(x.shape)
    for axis in reduction_axes:
        stats_shape[axis] = 1
    mean = mean.reshape(stats_shape)
    var = var.reshape(stats_shape)
    feature_shape = [1] * x.ndim
    reduced_feature_shape = []
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
        reduced_feature_shape.append(x.shape[ax])
    y = x - mean
    mul = lax.rsqrt(var + epsilon)
    args = [x]
    if use_scale:
        scale = mdl.param(
            'scale',
            nn.with_logical_partitioning(scale_init, ("embed",)),
            reduced_feature_shape,
            param_dtype,
        ).reshape(feature_shape)
        if nn.logical_to_mesh_axes(("embed",))[0] == "data":
            scale = partitioning_utils.with_logical_constraint(
                scale, P("embed",)
            )
        mul *= scale
        args.append(scale)
    y *= mul
    if use_bias:
        bias = mdl.param(
            'bias',
            nn.with_logical_partitioning(bias_init, ("embed",)),
            reduced_feature_shape,
            param_dtype,
        ).reshape(feature_shape)
        if nn.logical_to_mesh_axes(("embed",))[0] == "data":
            bias = partitioning_utils.with_logical_constraint(
                bias, P("embed",)
            )
        y += bias
        args.append(bias)
    dtype = canonicalize_dtype(*args, dtype=dtype)
    return jnp.asarray(y, dtype)


def _abs_sq(x):
    """Computes the elementwise square of the absolute value |x|^2."""
    if jnp.iscomplexobj(x):
        return lax.square(lax.real(x)) + lax.square(lax.imag(x))
    else:
        return lax.square(x)


def _compute_stats(
    x: Array,
    axes: Optional[Axes],
    dtype: Optional[Dtype],
    use_mean: bool = True,
):
    if dtype is None:
        dtype = jnp.result_type(x)
    # promote x to at least float32, this avoids half precision computation
    # but preserves double or complex floating points
    dtype = jnp.promote_types(dtype, jnp.float32)
    x = jnp.asarray(x, dtype)

    mean2 = jnp.mean(_abs_sq(x), axes)
    if use_mean:
        mean = jnp.mean(x, axes)
    else:
        mean = jnp.zeros(mean2.shape, dtype=dtype)

    # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
    # to floating point round-off errors.
    var = jnp.maximum(0., mean2 - _abs_sq(mean))
    return mean, var


class LayerNorm(nn.Module):
    """Layer normalization (https://arxiv.org/abs/1607.06450).
    LayerNorm normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within
    each example close to 0 and the activation standard deviation close to 1.
    Attributes:
        epsilon: A small float added to variance to avoid dividing by zero.
        dtype: the dtype of the result (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        use_bias:  If True, bias (beta) is added.
        use_scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
        bias_init: Initializer for bias, by default, zero.
        scale_init: Initializer for scale, by default, one.
        reduction_axes: Axes for computing normalization statistics.
        feature_axes: Feature axes for learned bias and scaling.
    """
    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.ones
    reduction_axes: Axes = -1
    feature_axes: Axes = -1

    @nn.compact
    def __call__(self, x):
        """Applies layer normalization on the input.
        Args:
        x: the inputs
        Returns:
        Normalized inputs (the same shape as inputs).
        """
        mean, var = _compute_stats(
            x, self.reduction_axes, self.dtype,
        )

        return _normalize(
            self,
            x,
            mean,
            var,
            self.reduction_axes,
            self.feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )
