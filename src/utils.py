import functools
import os
from typing import Any, Callable, Sequence, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
import ml_collections as mlc
from chex import Array
from jax.sharding import PartitionSpec

import flax
import flax.core.frozen_dict as frozen_dict
import flax.linen as nn
import flax.serialization as serialization
import flax.training.train_state as train_state

P = PartitionSpec


class TrainState(train_state.TrainState):
    dynamic_scale: Any
    dropout_rng: jnp.ndarray
    eval_apply_fn: Callable = flax.struct.field(pytree_node=False)
    generate_fn: Callable = flax.struct.field(pytree_node=False)


class DynamicScale(flax.struct.PyTreeNode):
    """Dynamic loss scaling for mixed precision gradients.
    For many models gradient computations in float16 will result in numerical
    issues because small/large gradients being flushed to zero/infinity.
    Dynamic loss scaling is an algorithm that aims to find the largest scalar
    multiple for which the gradient does not overflow. This way the risk of
    underflow is minimized.
    the `value_and_grad` method mimicks `jax.value_and_grad`. Beside the loss
    and gradients it also ouputs an updated `DynamicScale` instance with the
    current loss scale factor. This method also returns a boolean value indicating
    whether the gradients are finite.
    Example::
    from flax.training.dynamic_scale import DynamicScale
    def loss_fn(p):
        return jnp.asarray(p, jnp.float16) ** 2
    p = jnp.array(1., jnp.float32)
    dyn_scale = DynamicScale(growth_interval=10)
    compute_grad = jax.jit(lambda ds, p: ds.value_and_grad(loss_fn)(p))
    for _ in range(100):
        dyn_scale, is_fin, loss, grad = compute_grad(dyn_scale, p)
        p += jnp.where(is_fin, 0.01 * grad, 0.)
        print(loss)
    Jax currently cannot execute conditionals efficiently on GPUs therefore we
    selectifly ignore the gradient update using `jax.numpy.where` in case of
    non-finite gradients.
    Attributes:
    growth_factor: how much to grow the scalar after a period of finite
        gradients (default: 2.).
    backoff_factor: how much to shrink the scalar after a non-finite gradient
        (default: 0.5).
    growth_interval: after how many steps of finite gradients the scale should
        be increased (default: 2000).
    fin_steps: indicates how many gradient steps in a row have been finite.
    scale: the current scale by which the loss is multiplied.
    """

    growth_factor: float = flax.struct.field(pytree_node=False, default=2.0)
    backoff_factor: float = flax.struct.field(pytree_node=False, default=0.5)
    growth_interval: int = flax.struct.field(pytree_node=False, default=2000)
    fin_steps: Array = 0
    scale: Array = 65536.0

    def update(
        self,
        grad: frozen_dict.FrozenDict,
    ):
        finite = jax.tree_util.tree_reduce(
            lambda finite, g: finite & jnp.all(lax.is_finite(g)), grad, jnp.array(True)
        )

        grow = self.fin_steps == self.growth_interval
        fin_scale = jnp.where(
            grow & finite,
            jnp.minimum(self.scale * self.growth_factor, jnp.finfo(jnp.float32).max),
            self.scale,
        )
        inf_scale = self.scale * self.backoff_factor
        new_scale = jnp.where(finite, fin_scale, inf_scale)
        new_fin_steps = jnp.where(grow | (~finite), 0, self.fin_steps + 1)

        new_self = self.replace(fin_steps=new_fin_steps, scale=new_scale)
        return new_self, finite

    def value_and_grad(
        self,
        fun: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable:
        @functools.wraps(fun)
        def loss_wrapper(*args):
            aux = fun(*args)
            if has_aux:
                return (self.scale * aux[0], aux[1])
            else:
                return self.scale * aux

        grad_fn = jax.value_and_grad(loss_wrapper, argnums, has_aux)

        def grad_fn_wrapper(*args, weight=1.0):
            aux, grad = grad_fn(*args)
            aux = (aux[0] / self.scale, aux[1]) if has_aux else aux / self.scale

            grad = jax.tree_util.tree_map(
                lambda g: jnp.asarray(g, jnp.float32) / self.scale * weight, grad
            )

            return aux, grad

        return grad_fn_wrapper


class NoOp(flax.struct.PyTreeNode):
    scale: Array = 1.0

    def update(self, grad):
        return self, jnp.array(True)

    def value_and_grad(
        self,
        fun: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable:
        return jax.value_and_grad(fun, argnums, has_aux)


def setup_model(
    model_cls, pretrained_path, mp_num, from_pt, dtype, gradient_checkpointing
):
    with jax.default_device(jax.devices("cpu")[0]):
        model = model_cls.from_pretrained(pretrained_path, from_pt=from_pt, dtype=dtype)
        params = model.params
        original_vocab = model.config.vocab_size
        config = model.config

        if gradient_checkpointing:
            config.gradient_checkpointing = True

        if mp_num > 1:
            remainder = original_vocab % mp_num
            if remainder != 0:
                # deal with gpt2 vocab
                config.vocab_size = original_vocab + mp_num - remainder

                # expand embedding to be able to be partitioned
                emb = jnp.zeros((config.vocab_size, model.config.hidden_size))
                emb = emb.at[:original_vocab, :].set(
                    model.params["model"]["decoder"]["embed_tokens"]["embedding"].value
                )

                params["model"]["decoder"]["embed_tokens"][
                    "embedding"
                ] = nn.LogicallyPartitioned(
                    value=emb,
                    names=model.params["model"]["decoder"]["embed_tokens"][
                        "embedding"
                    ].names,
                )

        model = model_cls(config, _do_init=False, dtype=dtype)
        eval_model = (
            model
            if dtype == jnp.float32
            else model_cls(config, _do_init=False, dtype=jnp.float32)
        )

        if original_vocab != config.vocab_size:
            model.config.suppress_tokens += list(
                range(original_vocab, config.vocab_size)
            )
            eval_model.config.suppress_tokens += list(
                range(original_vocab, config.vocab_size)
            )

    return model, eval_model, params


def flatten_config(d):
    items = []
    for k, v in d.items():
        if isinstance(v, mlc.ConfigDict):
            items.extend(flatten_config(v).items())
        else:
            items.append((k, v))
    return dict(items)


def save_params(params: frozen_dict.FrozenDict, directory: str):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    with open(f"{directory}/params", "wb") as f:
        state_bytes = serialization.to_bytes(params)
        f.write(state_bytes)


def get_global_shape_dtypes(train_batch_size, eval_batch_size, data_args):
    if data_args.mode == "clm":
        block_size = data_args.block_size
        train_global_data_shape = {
            "input_ids": jax.ShapeDtypeStruct((train_batch_size, block_size), "i4"),
            "attention_mask": jax.ShapeDtypeStruct(
                (train_batch_size, block_size), "i4"
            ),
        }
        eval_global_data_shape = {
            "input_ids": jax.ShapeDtypeStruct((eval_batch_size, block_size), "i4"),
            "attention_mask": jax.ShapeDtypeStruct((eval_batch_size, block_size), "i4"),
        }
        axes = {
            "input_ids": P(
                "batch",
            ),
            "attention_mask": P(
                "batch",
            ),
        }
    elif data_args.mode == "seq2seq":
        input_ids_len = data_args.max_len
        if data_args.train.decoder_input_ids_column_name:
            decoder_ids_len = data_args.decoder_max_len
            train_global_data_shape = {
                "input_ids": jax.ShapeDtypeStruct(
                    (train_batch_size, input_ids_len), "i4"
                ),
                "attention_mask": jax.ShapeDtypeStruct(
                    (train_batch_size, input_ids_len), "i4"
                ),
                "decoder_input_ids": jax.ShapeDtypeStruct(
                    (train_batch_size, decoder_ids_len), "i4"
                ),
                "decoder_attention_mask": jax.ShapeDtypeStruct(
                    (train_batch_size, decoder_ids_len), "i4"
                ),
            }
            eval_global_data_shape = {
                "input_ids": jax.ShapeDtypeStruct(
                    (eval_batch_size, input_ids_len), "i4"
                ),
                "attention_mask": jax.ShapeDtypeStruct(
                    (eval_batch_size, input_ids_len), "i4"
                ),
                "decoder_input_ids": jax.ShapeDtypeStruct(
                    (eval_batch_size, decoder_ids_len), "i4"
                ),
                "decoder_attention_mask": jax.ShapeDtypeStruct(
                    (eval_batch_size, decoder_ids_len), "i4"
                ),
            }
            axes = {
                "input_ids": P(
                    "batch",
                ),
                "attention_mask": P(
                    "batch",
                ),
                "decoder_input_ids": P(
                    "batch",
                ),
                "decoder_attention_mask": P(
                    "batch",
                ),
            }
        else:
            train_global_data_shape = {
                "input_ids": jax.ShapeDtypeStruct(
                    (train_batch_size, input_ids_len), "i4"
                ),
                "attention_mask": jax.ShapeDtypeStruct(
                    (train_batch_size, input_ids_len), "i4"
                ),
            }
            eval_global_data_shape = {
                "input_ids": jax.ShapeDtypeStruct(
                    (eval_batch_size, input_ids_len), "i4"
                ),
                "attention_mask": jax.ShapeDtypeStruct(
                    (eval_batch_size, input_ids_len), "i4"
                ),
            }
            axes = {
                "input_ids": P(
                    "batch",
                ),
                "attention_mask": P(
                    "batch",
                ),
            }
    else:
        raise NotImplementedError("Mode can either be seq2seq or clm")

    return train_global_data_shape, eval_global_data_shape, axes
