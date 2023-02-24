from functools import partial
from typing import Any, Callable, Optional, Tuple
import dataclasses

import flax.linen as nn
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
import jax
from jax import lax
import jax.experimental.shard_map as shard_map
import jax.random as jrandom
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding

P = PartitionSpec

import numpy as np


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


def dot_gather_accum(kernel, inputs):
    axis_size = lax.psum(1, axis_name="data")
    axis_index = lax.axis_index(axis_name="data")
    chunk_size = kernel.shape[0]

    def f(i, carrys):
        accum, kernel = carrys
        x = lax.dynamic_slice(
            inputs,
            (0, 0, ((axis_index + i) % axis_size) * chunk_size),
            (inputs.shape[0], inputs.shape[0], chunk_size),
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

    x = lax.dynamic_slice(
        inputs,
        (0, 0, ((axis_index + i) % axis_size) * chunk_size),
        (inputs.shape[0], inputs.shape[0], chunk_size),
    )
    update = jnp.dot(x, kernel)

    accum = accum + update

    return accum


def add_bias(x, y):
    x = lax.all_gather(x, "data", axis=0, tiled=True)
    x = jnp.reshape(x, (1,) * (y.ndim - 1) + (-1,))
    return x + y


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
                dot_gather_0,
                in_specs=(
                    P("data"),
                    P("data"),
                ),
                out_specs=P("data"),
                mesh=self.mesh,
            )
        elif mesh_axes[1] == "data":
            dot_fn = shard_map.shard_map(
                dot_gather_1,
                in_specs=(
                    P(None, "data"),
                    P("data"),
                ),
                out_specs=P("data"),
                mesh=self.mesh,
            )
        else:
            dot_fn = jnp.dot

        y = dot_fn(kernel, inputs)

        if bias is not None:
            if mesh_axes[1] == "data":
                y = shard_map.shard_map(
                    add_bias,
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


def take_gather_1(embedding, inputs):
    gathered = lax.all_gather(
        embedding,
        "data",
        axis=1,
        tiled=True,
    )
    inputs = jax.nn.one_hot(inputs, gathered.shape[0])

    return jnp.dot(inputs, gathered)


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
                take_gather_0,
                in_specs=(P("data"), P("data")),
                out_specs=P("data"),
                mesh=self.mesh,
                check_rep=False,
            )
        elif mesh_axes[1] == "data":
            self.embed_fn = shard_map.shard_map(
                take_gather_1,
                in_specs=(P(None, "data"), P("data")),
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


if __name__ == "__main__":
    import flax.training.train_state as train_state
    import optax
    from flax.core.meta import Partitioned

    TrainState = train_state.TrainState

    mesh = Mesh(np.array(jax.devices()).reshape(8, 1), ("data", "model"))

    nn.set_logical_axis_rules(
        (
            ("batch", "data"),
            ("shard", "data"),
            ("no_shard", None),
        )
    )

    model = Dense(
        128,
        use_bias=False,
        kernel_init=default_kernel_init,
        names=("shard", "no_shard"),
    )

    rng = jrandom.PRNGKey(2)

    def init_fn(x):
        return model.init(rng, x)["params"]

    param_shapes = jax.eval_shape(init_fn, jnp.ones((8, 64, 256)))
    mesh_param_spec = jax.tree_util.tree_map(
        lambda pspec: NamedSharding(mesh, pspec),
        nn.logical_to_mesh(nn.get_partition_spec(param_shapes)),
    )

    params = jax.jit(init_fn, out_shardings=mesh_param_spec)(jnp.ones((8, 64, 256)))

    def apply_fn(params, x):
        return model.apply({"params": params}, x)

    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=1e-4,
        transition_steps=50,
    )
    decay_fn = optax.linear_schedule(
        init_value=1e-4,
        end_value=0.0,
        transition_steps=100 - 50,
    )
    last_boundary = 50
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[last_boundary],
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule_fn),
    )

    def create_state(params):
        return TrainState.create(params=params, tx=tx, apply_fn=apply_fn)

    def get_partition_spec(tree):
        def f(x):
            if isinstance(x, Partitioned):
                return x.get_partition_spec()
            else:
                return P()

        return jax.tree_map(f, tree, is_leaf=lambda x: isinstance(x, Partitioned))

    train_state_shapes = jax.eval_shape(create_state, params)
    mesh_train_state_spec = jax.tree_util.tree_map(
        lambda pspec: NamedSharding(mesh, pspec),
        nn.logical_to_mesh(get_partition_spec(train_state_shapes)),
    )
    mesh_param_spec = mesh_train_state_spec.params

    def create_fn(params):
        params = jax.lax.with_sharding_constraint(params, mesh_param_spec)
        return create_state(params)

    train_state = jax.jit(create_fn, out_shardings=mesh_train_state_spec)(params)

    def compute_loss(params, x, y):
        out = apply_fn(params, x)
        loss = (y - out) ** 2
        return jnp.mean(loss)

    grad_fn = jax.value_and_grad(compute_loss)

    @partial(
        jax.jit,
        out_shardings=(mesh_train_state_spec, NamedSharding(mesh, P())),
        donate_argnums=(0,),
    )
    def train_step(train_state, batch):
        with jax.named_scope("grads"):
            loss, grads = grad_fn(
                lax.with_sharding_constraint(train_state.params, mesh_param_spec),
                batch["x"],
                batch["y"],
            )

        grads = lax.with_sharding_constraint(grads, mesh_param_spec)

        with jax.named_scope("update"):
            new_train_state = train_state.apply_gradients(grads=grads)

        return new_train_state, loss

    import time

    x = jax.device_put(jnp.ones((8, 64, 256)), NamedSharding(mesh, P("data")))
    y = jax.device_put(
        jrandom.uniform(rng, (8, 64, 128)), NamedSharding(mesh, P("data"))
    )

    times = []

    train_state, l = train_step(train_state, {"x": x, "y": y})

    for i in range(100):
        t = time.time()
        train_state, l = train_step(train_state, {"x": x, "y": y})
        t1 = time.time()
        times.append(t1 - t)

    print(np.mean(times))
