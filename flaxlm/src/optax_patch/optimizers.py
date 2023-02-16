"""Optax implementation of the Lion optimizer."""

from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
import optax

class ScaleByScheduleState(NamedTuple):
    """Maintains count for scale scheduling."""
    count: chex.Array  # shape=(1), dtype=jnp.int32

def scale_by_schedule(
    step_size_fn: optax.Schedule
) -> optax.GradientTransformation:
    """Scale updates using a custom schedule for the `step_size`.
    Args:
    step_size_fn: A function that takes an update count as input and proposes
      the step_size to multiply the updates by.
    Returns:
    A `GradientTransformation` object.
    """

    def init_fn(params):
        del params
        return ScaleByScheduleState(count=jnp.zeros(1, jnp.int32))

    def update_fn(updates, state, params=None):
        del params
        step_size = step_size_fn(state.count)
        updates = jax.tree_util.tree_map(
            lambda g: jnp.array(step_size, dtype=g.dtype) * g, updates)
        return updates, ScaleByScheduleState(
            count=optax.safe_int32_increment(state.count)
        )

    return optax.GradientTransformation(init_fn, update_fn)


def _scale_by_learning_rate(
    learning_rate, flip_sign=True
):
    m = -1 if flip_sign else 1
    if callable(learning_rate):
        return scale_by_schedule(lambda count: m * learning_rate(count))
    return optax.scale(m * learning_rate)


def update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order`-th moment."""
    return jax.tree_util.tree_map(
          lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments
    )


class ScaleByLionState(NamedTuple):
    """State for the Lion algorithm."""
    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: optax.Updates


def scale_by_lion(
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
    """Rescale updates according to the Lion algorithm.
    Args:
    b1: rate for combining moment and the current grad.
    b2: decay rate for the exponentially weighted average of grads.
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype is inferred from `params` and `updates`.
    Returns:
    A `GradientTransformation` object.
    """

    def init_fn(params):
        mu = jax.tree_util.tree_map(  # moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        return ScaleByLionState(count=jnp.zeros(1, jnp.int32), mu=mu)

    def update_fn(updates, state, params=None):
        del params
        mu = update_moment(updates, state.mu, b2, 1)
        mu = jax.tree_map(lambda x: x.astype(mu_dtype), mu)
        count_inc = optax.safe_int32_increment(state.count)
        updates = jax.tree_util.tree_map(
            lambda g, m: jnp.sign((1. - b1) * g + b1 * m), updates, state.mu
        )
        return updates, ScaleByLionState(count=count_inc, mu=mu)

    return optax.GradientTransformation(init_fn, update_fn)


def lion(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[optax.Params], Any]]] = None,
) -> optax.GradientTransformation:
    """Lion.
    Args:
    learning_rate: A fixed global scaling factor.
    b1: Exponential decay rate to combine the gradient and the moment.
    b2: Exponential decay rate to track the moment of past gradients.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Adam gradient transformations are applied to all parameters.
    Returns:
    The corresponding `GradientTransformation`.
    """
    return optax.chain(
        scale_by_lion(b1=b1, b2=b2, mu_dtype=mu_dtype),
        optax.add_decayed_weights(weight_decay, mask),
        _scale_by_learning_rate(learning_rate),
    )


def adamw(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[optax.Params], Any]]] = None,
) -> optax.GradientTransformation:
  """Adam with weight decay regularization.
  AdamW uses weight decay to regularize learning towards small weights, as
  this leads to better generalization. In SGD you can also use L2 regularization
  to implement this as an additive loss term, however L2 regularization
  does not behave as intended for adaptive gradient algorithms such as Adam.
  References:
    Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101
  Args:
    learning_rate: A fixed global scaling factor.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Adam gradient transformations are applied to all parameters.
  Returns:
    The corresponding `GradientTransformation`.
  """
  return optax.chain(
      optax.scale_by_adam(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
      optax.add_decayed_weights(weight_decay, mask),
      _scale_by_learning_rate(learning_rate),
  )
