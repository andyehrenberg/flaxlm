# coding=utf-8
# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Flax whisper model."""

import random
import warnings
from functools import partial
from typing import Dict, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from transformers.generation.flax_logits_process import (
    FlaxForcedBOSTokenLogitsProcessor, FlaxForcedEOSTokenLogitsProcessor,
    FlaxLogitsProcessor, FlaxLogitsProcessorList, FlaxMinLengthLogitsProcessor)
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutput, FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions, FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput)
from transformers.modeling_flax_utils import (ACT2FN, FlaxPreTrainedModel,
                                              append_call_sample_docstring,
                                              append_replace_return_docstrings,
                                              overwrite_call_docstring)
from transformers.utils import (add_start_docstrings,
                                add_start_docstrings_to_model_forward, logging,
                                replace_return_docstrings)

from src.transformers_patch.whisper_config_remat import WhisperConfig

WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
FLAX_WEIGHTS_INDEX_NAME = "flax_model.msgpack.index.json"

import os
from pickle import UnpicklingError

import msgpack.exceptions
from flax.serialization import from_bytes
from transformers import PretrainedConfig
from transformers.utils import (cached_file, download_url, has_file,
                                is_remote_url)
from transformers.utils.hub import get_checkpoint_shard_files

logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "openai/whisper-tiny"
_CONFIG_FOR_DOC = "WhisperConfig"
_TOKENIZER_FOR_DOC = "WhisperTokenizer"


WHISPER_START_DOCSTRING = r"""
    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.) This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.
    Finally, this model supports inherent JAX features such as:
    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
    Parameters:
        config ([`WhisperConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs). This can be used to enable mixed-precision training or half-precision
            inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.** If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`]
            and [`~FlaxPreTrainedModel.to_bf16`].
"""

WHISPER_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`numpy.ndarray` of shape `(batch_size, feature_size, sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`WhisperFeatureExtractor`] should be used for extracting the features, padding and conversion into a
            tensor of type `numpy.ndarray`. See [`~WhisperFeatureExtractor.__call__`]
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Whisper does not support masking of the `input_features`, this argument is preserved for compatibility, but
            is not used. By default the silence in the input log mel spectrogram are ignored.
        decoder_input_ids (`numpy.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using
            [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.
            [What are decoder input IDs?](../glossary#decoder-input-ids) Whisper uses the `decoder_start_token_id` as
            the starting token for `decoder_input_ids` generation.
        decoder_attention_mask (`numpy.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default. If you want to change padding behavior, you should modify to your needs. See diagram 1
            in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Whisper does not use position_ids in the encoder as input_features is always the same size and doesn't use
            masking, but this argument is preserved for compatibility. By default the silence in the input log mel
            spectrogram are ignored.
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
            range `[0, config.max_position_embeddings - 1]`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

WHISPER_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`numpy.ndarray` of shape `(batch_size, feature_size, sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`WhisperFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `numpy.ndarray`. See [`~WhisperFeatureExtractor.__call__`].
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Whisper does not support masking of the `input_features`, this argument is preserved for compatibility, but
            is not used. By default the silence in the input log mel spectrogram are ignored.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

WHISPER_DECODE_INPUTS_DOCSTRING = r"""
    Args:
        decoder_input_ids (`numpy.ndarray` of shape `(batch_size, target_sequence_length)`):
            Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using
            [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.
            [What are decoder input IDs?](../glossary#decoder-input-ids)
        encoder_outputs (`tuple(tuple(numpy.ndarray)`):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        encoder_attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
           Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
            but it is not used. By default the silence in the input log mel spectrogram are ignored.
        decoder_attention_mask (`numpy.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default. If you want to change padding behavior, you should modify to your needs. See diagram 1
            in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
            range `[0, config.max_position_embeddings - 1]`.
        past_key_values (`Dict[str, numpy.ndarray]`, *optional*, returned by `init_cache` or when passing previous
        `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class FlaxSuppressTokensLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] suppressing a list of tokens at each decoding step. The processor will set their log probs
    to be `-inf` so they are not sampled.

    Args:
        suppress_tokens (`list`):
            Tokens to not sample.
    """

    def __init__(self, suppress_tokens: list):
        self.suppress_tokens = list(suppress_tokens)

    def __call__(
        self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int
    ) -> jnp.ndarray:
        scores = scores.at[..., self.suppress_tokens].set(-float("inf"))

        return scores


class FlaxSuppressTokensAtBeginLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] supressing a list of tokens as soon as the `generate` function starts generating using
    `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` are not sampled at the
    begining of the generation.

    Args:
        begin_suppress_tokens (`List[int]`):
            Tokens to not sample.
        begin_index (`int`):
            Index where the tokens are suppressed.
    """

    def __init__(self, begin_suppress_tokens, begin_index):
        self.begin_suppress_tokens = list(begin_suppress_tokens)
        self.begin_index = begin_index

    def __call__(self, input_ids, scores, cur_len: int):
        apply_penalty = 1 - jnp.bool_(cur_len - self.begin_index)

        scores = jnp.where(
            apply_penalty,
            scores.at[:, self.begin_suppress_tokens].set(-float("inf")),
            scores,
        )

        return scores


class FlaxForceTokenAtIdxLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] forcing a token to be sampled at an index.

    Args:
        apply_idx (`int`):
            Index where sampling is forced.
        token_id (`int`):
            Token that is forced to be sampled.
    """

    def __init__(self, apply_idx: int, token_id: int):
        self.apply_idx = apply_idx
        self.token_id = token_id

    def __call__(
        self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int
    ) -> jnp.ndarray:
        new_scores = jnp.full(scores.shape, -float("inf"))

        apply_penalty = 1 - jnp.bool_(cur_len - self.apply_idx)

        scores = jnp.where(
            apply_penalty, new_scores.at[:, self.token_id].set(0), scores
        )

        return scores


class FlaxForceTokensLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that takes a list of pairs of integers which indicates a mapping from generation indices to
    token indices that will be forced before sampling. The processor will set their log probs to `inf` so that they are
    sampled at their corresponding index.

    Args:
        force_token_map (`list`):
            Map giving token ids and indices where they will be forced to be sampled.
    """

    def __init__(self, force_token_map):
        self.processors = [
            FlaxForceTokenAtIdxLogitsProcessor(i[0], i[1]) for i in force_token_map
        ]

    def __call__(
        self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int
    ) -> jnp.ndarray:
        for processor in self.processors:
            scores = processor(input_ids, scores, cur_len)

        return scores


class FlaxWhisperAttention(nn.Module):
    config: WhisperConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                jax.nn.initializers.normal(self.config.init_std),
                ("embed", "joined_kv"),
            ),
            bias_init=nn.with_logical_partitioning(
                jax.nn.initializers.zeros, ("joined_kv",)
            ),
        )

        self.q_proj = dense()
        self.k_proj = dense(use_bias=False)
        self.v_proj = dense()
        self.out_proj = nn.Dense(
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                jax.nn.initializers.normal(self.config.init_std),
                ("joined_kv", "embed"),
            ),
            bias_init=nn.with_logical_partitioning(
                jax.nn.initializers.zeros, ("embed",)
            ),
        )

        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_target_positions), dtype="bool"),
                dtype="bool",
            )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        is_cross_attention = key_value_states is not None
        batch_size = hidden_states.shape[0]

        q = self.q_proj(hidden_states)

        if is_cross_attention:
            k = self.k_proj(key_value_states)
            v = self.v_proj(key_value_states)
        else:
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        q, k, v = jax.tree_util.tree_map(self._split_heads, (q, k, v))

        if self.causal:
            query_length, key_length = q.shape[1], k.shape[1]
            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_mask = lax.dynamic_slice(
                    self.causal_mask,
                    (0, 0, mask_shift, 0),
                    (1, 1, query_length, max_decoder_length),
                )
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]
            causal_mask = jnp.broadcast_to(
                causal_mask, (batch_size,) + causal_mask.shape[1:]
            )

        # combine masks if needed
        if attention_mask is not None and self.causal:
            attention_mask = jnp.broadcast_to(
                jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
            )
            attention_mask = combine_masks(attention_mask, causal_mask)
        elif self.causal:
            attention_mask = causal_mask
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.

        if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
            k, v, attention_mask = self._concatenate_to_cache(k, v, q, attention_mask)

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            # attention mask in the form of attention bias
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0),
                jnp.full(attention_mask.shape, -1e4),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            q,
            k,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, v)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

    def _split_heads(self, hidden_state) -> jnp.ndarray:
        hidden_state = hidden_state.reshape(
            hidden_state.shape[:2] + (self.num_heads, self.head_dim)
        )
        hidden_state = nn.with_logical_constraint(
            hidden_state, ("batch", "length", "heads", "kv")
        )
        return hidden_state

    def _merge_heads(self, hidden_state) -> jnp.ndarray:
        return hidden_state.reshape(hidden_state.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(
        self, key, value, query, attention_mask
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable(
            "cache", "cached_key", jnp.zeros, key.shape, key.dtype
        )
        cached_value = self.variable(
            "cache", "cached_value", jnp.zeros, value.shape, value.dtype
        )
        cache_index = self.variable(
            "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
        )

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only
            # attend to those key positions that have already been generated and cached, not the
            # remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)

        return key, value, attention_mask


class FlaxWhisperEncoderLayer(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.embed_dim = self.config.d_model
        self.self_attn = FlaxWhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        self.self_attn_layer_norm = nn.LayerNorm(
            dtype=self.dtype,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("embed",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("embed",)),
            epsilon=1e-5,
        )
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                jax.nn.initializers.normal(self.config.init_std),
                ("embed", "mlp"),
            ),
            bias_init=nn.with_logical_partitioning(jax.nn.initializers.zeros, ("mlp",)),
        )
        self.fc2 = nn.Dense(
            self.embed_dim,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                jax.nn.initializers.normal(self.config.init_std),
                ("mlp", "embed"),
            ),
            bias_init=nn.with_logical_partitioning(
                jax.nn.initializers.zeros, ("embed",)
            ),
        )
        self.final_layer_norm = nn.LayerNorm(
            dtype=self.dtype,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("embed",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("embed",)),
            epsilon=1e-5,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout_layer(
            hidden_states, deterministic=deterministic
        )
        hidden_states = nn.with_logical_constraint(
            hidden_states, ("batch", "length", "mlp")
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class FlaxWhisperEncoderLayerCollection(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        layer = (
            nn_partitioning.remat(
                FlaxWhisperEncoderLayer,
                static_argnums=(2, 3),
            )
            if self.config.gradient_checkpointing
            else FlaxWhisperEncoderLayer
        )
        self.layers = [
            layer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.encoder_layers)
        ]
        self.layerdrop = self.config.encoder_layerdrop

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not deterministic and (
                dropout_probability < self.layerdrop
            ):  # skip the layer
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    deterministic,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states, all_hidden_states, all_attentions)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class FlaxWhisperDecoderLayer(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.embed_dim = self.config.d_model
        self.self_attn = FlaxWhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
        )
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        self.self_attn_layer_norm = nn.LayerNorm(
            dtype=self.dtype,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("embed",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("embed",)),
            epsilon=1e-5,
        )
        self.encoder_attn = FlaxWhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(
            dtype=self.dtype,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("embed",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("embed",)),
            epsilon=1e-5,
        )
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                jax.nn.initializers.normal(self.config.init_std),
                ("embed", "mlp"),
            ),
            bias_init=nn.with_logical_partitioning(jax.nn.initializers.zeros, ("mlp",)),
        )
        self.fc2 = nn.Dense(
            self.embed_dim,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                jax.nn.initializers.normal(self.config.init_std),
                ("embed", "mlp"),
            ),
            bias_init=nn.with_logical_partitioning(jax.nn.initializers.zeros, ("mlp",)),
        )
        self.final_layer_norm = nn.LayerNorm(
            dtype=self.dtype,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("embed",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("embed",)),
            epsilon=1e-5,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            init_cache=init_cache,
        )
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = self.dropout_layer(
                hidden_states, deterministic=deterministic
            )
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout_layer(
            hidden_states, deterministic=deterministic
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class FlaxWhisperDecoderLayerCollection(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        layer = (
            nn_partitioning.remat(
                FlaxWhisperDecoderLayer,
                static_argnums=(4, 5, 6),
            )
            if self.config.gradient_checkpointing
            else FlaxWhisperDecoderLayer
        )
        self.layers = [
            layer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.decoder_layers)
        ]
        self.layerdrop = self.config.decoder_layerdrop

    def __call__(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None, None)
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    init_cache,
                    output_attentions,
                    deterministic,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = [
            hidden_states,
            all_hidden_states,
            all_self_attns,
            all_cross_attentions,
        ]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class FlaxWhisperEncoder(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.conv1 = nn.Conv(
            self.config.d_model,
            kernel_size=(3,),
            padding=1,
            kernel_init=nn.with_logical_partitioning(
                jax.nn.initializers.normal(self.config.init_std),
                ("conv_in", "conv_out"),
            ),
            bias_init=nn.with_logical_partitioning(
                jax.nn.initializers.zeros, ("conv_out",)
            ),
            dtype=self.dtype,
        )
        self.conv2 = nn.Conv(
            self.config.d_model,
            kernel_size=(3,),
            strides=2,
            padding=1,
            kernel_init=nn.with_logical_partitioning(
                jax.nn.initializers.normal(self.config.init_std),
                ("conv_in", "conv_out"),
            ),
            bias_init=nn.with_logical_partitioning(
                jax.nn.initializers.zeros, ("conv_out",)
            ),
            dtype=self.dtype,
        )

        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        self.layers = FlaxWhisperEncoderLayerCollection(
            self.config,
            dtype=self.dtype,
        )
        self.embed_positions = nn.Embed(
            self.config.max_source_positions,
            self.config.d_model,
            embedding_init=nn.with_logical_partitioning(
                nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
                ("vocab", "embed"),
            ),
            dtype=self.dtype,
        )

        self.layer_norm = nn.LayerNorm(
            dtype=self.dtype,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("embed",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("embed",)),
            epsilon=1e-5,
        )

    def __call__(
        self,
        input_features: jnp.ndarray,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        if input_features.shape[1:] != (
            self.config.num_mel_bins,
            self.config.max_source_positions * 2,
        ):
            raise ValueError(
                "input_features.shape[1:], must be equal to (self.config.num_mel_bins,"
                f" self.config.max_source_positions * 2) (got {input_features.shape[1:]}, but should be"
                f" ({self.config.num_mel_bins}, {self.config.max_source_positions * 2}))"
            )

        input_features = input_features.transpose(0, 2, 1)
        hidden_states = jax.nn.gelu(self.conv1(input_features), approximate=False)
        hidden_states = jax.nn.gelu(self.conv2(hidden_states), approximate=False)

        embed_positions = self.embed_positions(
            jnp.arange(self.config.max_source_positions)
        )
        embed_positions = nn.with_logical_constraint(
            embed_positions, ("length", "embed")
        )

        hidden_states = hidden_states + embed_positions

        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        outputs = self.layers(
            hidden_states,
            attention_mask=None,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = self.layer_norm(outputs[0])

        if not return_dict:
            return (hidden_states,) + outputs[1:]

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxWhisperDecoder(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=nn.with_logical_partitioning(
                nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
                ("vocab", "embed"),
            ),
            dtype=self.dtype,
        )
        self.embed_positions = nn.Embed(
            self.config.max_target_positions,
            self.config.d_model,
            embedding_init=nn.with_logical_partitioning(
                nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
                ("vocab", "embed"),
            ),
            dtype=self.dtype,
        )

        self.layers = FlaxWhisperDecoderLayerCollection(self.config, dtype=self.dtype)

        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        self.layer_norm = nn.LayerNorm(
            dtype=self.dtype,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("embed",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("embed",)),
            epsilon=1e-5,
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        input_embeds = self.embed_tokens(input_ids)
        input_embeds = nn.with_logical_constraint(
            input_embeds, ("batch", "length", "embed")
        )
        position_embeds = self.embed_positions(position_ids)
        position_embeds = nn.with_logical_constraint(
            position_embeds, ("batch", "length", "embed")
        )

        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        outputs = self.layers(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_states = outputs[0]
        last_hidden_states = self.layer_norm(last_hidden_states)

        # update the last element in `hidden_states` after applying `layernorm` above
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)

        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (
                outputs[2:] if output_hidden_states else outputs[1:]
            )
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class FlaxWhisperModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.encoder = FlaxWhisperEncoder(self.config, dtype=self.dtype)
        self.decoder = FlaxWhisperDecoder(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_features: jnp.ndarray,
        decoder_input_ids: jnp.ndarray,
        decoder_attention_mask: jnp.ndarray,
        decoder_position_ids: jnp.ndarray,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        encoder_outputs = self.encoder(
            input_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder


class FlaxWhisperPreTrainedModel(FlaxPreTrainedModel):
    config_class = WhisperConfig
    base_model_prefix: str = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: WhisperConfig,
        input_shape: Tuple[int] = (1, 80, 3000),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None
    ) -> FrozenDict:
        # init input tensors
        input_features = jnp.zeros(input_shape)
        input_features = input_features.at[(..., -1)].set(self.config.eos_token_id)

        decoder_input_ids = jnp.zeros((input_shape[0], 1), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        batch_size, sequence_length = decoder_input_ids.shape
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
        )

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(
            rngs,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
        )["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        dtype: jnp.dtype = jnp.float32,
        *model_args,
        **kwargs,
    ):
        config = kwargs.pop("config", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_pt = kwargs.pop("from_pt", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _do_init = kwargs.pop("_do_init", True)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)

        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )

        user_agent = {
            "file_type": "model",
            "framework": "flax",
            "from_auto_class": from_auto_class,
        }
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = (
                config if config is not None else pretrained_model_name_or_path
            )
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                _commit_hash=commit_hash,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        if commit_hash is None:
            commit_hash = getattr(config, "_commit_hash", None)

        # Add the dtype to model_kwargs
        model_kwargs["dtype"] = dtype

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False

        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if os.path.isdir(pretrained_model_name_or_path):
                if from_pt and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, WEIGHTS_NAME
                    )
                elif from_pt and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, subfolder, WEIGHTS_INDEX_NAME
                    )
                ):
                    # Load from a sharded pytorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, WEIGHTS_INDEX_NAME
                    )
                    is_sharded = True
                elif os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME
                    )
                ):
                    # Load from a Flax checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME
                    )
                elif os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        FLAX_WEIGHTS_INDEX_NAME,
                    )
                ):
                    # Load from a sharded Flax checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        FLAX_WEIGHTS_INDEX_NAME,
                    )
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)
                ):
                    raise EnvironmentError(
                        f"Error no file named {FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} "
                        "but there is a file for PyTorch weights. Use `from_pt=True` to load this model from those "
                        "weights."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME} found in directory "
                        f"{pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                filename = WEIGHTS_NAME if from_pt else FLAX_WEIGHTS_NAME
                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = dict(
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        user_agent=user_agent,
                        revision=revision,
                        subfolder=subfolder,
                        _raise_exceptions_for_missing_entries=False,
                        _commit_hash=commit_hash,
                    )
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path, filename, **cached_file_kwargs
                    )

                    # Since we set _raise_exceptions_for_missing_entries=False, we don't get an expection but a None
                    # result when internet is up, the repo and revision exist, but the file does not.
                    if resolved_archive_file is None and filename == FLAX_WEIGHTS_NAME:
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            FLAX_WEIGHTS_INDEX_NAME,
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    # Maybe the checkpoint is pytorch sharded, we try to grab the pytorch index name in this case.
                    elif resolved_archive_file is None and from_pt:
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            WEIGHTS_INDEX_NAME,
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None:
                        # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                        # message.
                        has_file_kwargs = {
                            "revision": revision,
                            "proxies": proxies,
                            "use_auth_token": use_auth_token,
                        }
                        if has_file(
                            pretrained_model_name_or_path,
                            WEIGHTS_NAME,
                            **has_file_kwargs,
                        ):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {FLAX_WEIGHTS_NAME} but there is a file for PyTorch weights. Use `from_pt=True` to"
                                " load this model from those weights."
                            )
                        elif has_file(
                            pretrained_model_name_or_path,
                            WEIGHTS_INDEX_NAME,
                            **has_file_kwargs,
                        ):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {FLAX_WEIGHTS_INDEX_NAME} but there is a sharded file for PyTorch weights. Use"
                                " `from_pt=True` to load this model from those weights."
                            )
                        else:
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME}."
                            )
                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception:
                    # For any other exception, we throw a generic error.
                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME}."
                    )

            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(
                    f"loading weights file {filename} from cache at {resolved_archive_file}"
                )
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, _ = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=commit_hash,
            )

        # init random models
        model = cls(config, *model_args, _do_init=_do_init, **model_kwargs)

        if from_pt:
            state = load_pytorch_checkpoint_in_flax_state_dict(
                model, resolved_archive_file, is_sharded
            )
        else:

            if is_sharded:
                state = cls.load_flax_sharded_weights(resolved_archive_file)
            else:
                try:
                    with open(resolved_archive_file, "rb") as state_f:
                        state = from_bytes(cls, state_f.read())
                except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
                    try:
                        with open(resolved_archive_file) as f:
                            if f.read().startswith("version"):
                                raise OSError(
                                    "You seem to have cloned a repository without having git-lfs installed. Please"
                                    " install git-lfs and run `git lfs install` followed by `git lfs pull` in the"
                                    " folder you cloned."
                                )
                            else:
                                raise ValueError from e
                    except (UnicodeDecodeError, ValueError):
                        raise EnvironmentError(
                            f"Unable to convert {archive_file} to Flax deserializable object. "
                        )
            # make sure all arrays are stored as jnp.arrays
            # NOTE: This is to prevent a bug this will be fixed in Flax >= v0.3.4:
            # https://github.com/google/flax/issues/1261
            if _do_init:
                state = jax.tree_util.tree_map(jnp.array, state)
            else:
                # keep the params on CPU if we don't want to initialize
                state = jax.tree_util.tree_map(
                    lambda x: jax.device_put(x, jax.devices("cpu")[0]), state
                )

        # if model is base model only use model_prefix key
        if (
            cls.base_model_prefix not in dict(model.params_shape_tree)
            and cls.base_model_prefix in state
        ):
            state = state[cls.base_model_prefix]

        # if model is head model and we are loading weights from base model
        # we initialize new params dict with base_model_prefix
        if (
            cls.base_model_prefix in dict(model.params_shape_tree)
            and cls.base_model_prefix not in state
        ):
            state = {cls.base_model_prefix: state}

        # flatten dicts
        state = flatten_dict(state)

        random_state = flatten_dict(
            unfreeze(model.params if _do_init else model.params_shape_tree)
        )

        missing_keys = model.required_params - set(state.keys())
        unexpected_keys = set(state.keys()) - model.required_params

        if missing_keys and not _do_init:
            logger.warning(
                f"The checkpoint {pretrained_model_name_or_path} is missing required keys: {missing_keys}. "
                "Make sure to call model.init_weights to initialize the missing weights."
            )
            cls._missing_keys = missing_keys

        # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
        # matching the weights in the model.
        mismatched_keys = []
        for key in state.keys():
            state_key = state[key].value if hasattr(state[key], "value") else state[key]
            random_state_key = (
                random_state[key].value
                if hasattr(random_state[key], "value")
                else random_state[key]
            )
            if key in random_state and state_key.shape != random_state_key.shape:
                if ignore_mismatched_sizes:
                    mismatched_keys.append(
                        (key, state_key.shape, random_state[key].shape)
                    )
                    state_key = random_state_key
                else:
                    raise ValueError(
                        f"Trying to load the pretrained weight for {key} failed: checkpoint has shape "
                        f"{state_key.shape} which is incompatible with the model shape {random_state_key.shape}. "
                        "Using `ignore_mismatched_sizes=True` if you really want to load this checkpoint inside this "
                        "model."
                    )

        # add missing keys as random parameters if we are initializing
        if missing_keys and _do_init:
            for missing_key in missing_keys:
                state[missing_key] = random_state[missing_key]

        # remove unexpected keys to not be saved again
        for unexpected_key in unexpected_keys:
            del state[unexpected_key]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(
                f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n"
            )

        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        # dictionary of key: dtypes for the model params
        param_dtypes = jax.tree_util.tree_map(lambda x: x.dtype, state)
        # extract keys of parameters not in jnp.float32
        fp16_params = [k for k in param_dtypes if param_dtypes[k] == jnp.float16]
        bf16_params = [k for k in param_dtypes if param_dtypes[k] == jnp.bfloat16]

        # raise a warning if any of the parameters are not in jnp.float32
        if len(fp16_params) > 0:
            logger.warning(
                f"Some of the weights of {model.__class__.__name__} were initialized in float16 precision from "
                f"the model checkpoint at {pretrained_model_name_or_path}:\n{fp16_params}\n"
                "You should probably UPCAST the model weights to float32 if this was not intended. "
                "See [`~FlaxPreTrainedModel.to_fp32`] for further information on how to do this."
            )

        if len(bf16_params) > 0:
            logger.warning(
                f"Some of the weights of {model.__class__.__name__} were initialized in bfloat16 precision from "
                f"the model checkpoint at {pretrained_model_name_or_path}:\n{bf16_params}\n"
                "You should probably UPCAST the model weights to float32 if this was not intended. "
                "See [`~FlaxPreTrainedModel.to_fp32`] for further information on how to do this."
            )

        if _do_init:
            # set correct parameters
            model.params = unflatten_dict(state)
            return model
        else:
            return model, unflatten_dict(state)

    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*:
                `attentions`). `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*)
                is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross-attention of the decoder.
        """
        # init input variables to retrieve cache
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]),
            decoder_input_ids.shape,
        )

        def _decoder_forward(
            module,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
            **kwargs,
        ):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
                **kwargs,
            )

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # we only need to call the decoder to init the cache
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(WHISPER_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutput, config_class=WhisperConfig
    )
    def encode(
        self,
        input_features: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        **kwargs,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
        >>> input_features = inputs.input_features
        >>> encoder_outputs = model.encode(input_features=input_features)
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _encoder_forward(module, input_features, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_features, **kwargs)

        return self.module.apply(
            {"params": params or self.params},
            input_features=jnp.array(input_features, dtype="f4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            method=_encoder_forward,
        )

    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutputWithPastAndCrossAttentions,
        config_class=WhisperConfig,
    )
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
        >>> input_features = inputs.input_features
        >>> encoder_outputs = model.encode(input_features=input_features)
        >>> decoder_start_token_id = model.config.decoder_start_token_id

        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> last_decoder_hidden_states = outputs.last_hidden_state
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        encoder_hidden_states = encoder_outputs[0]

        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `decoder_position_ids` when passing `past_key_values`."
                )

            if decoder_attention_mask is not None:
                decoder_position_ids = (
                    decoder_attention_mask.cumsum(axis=-1) * decoder_attention_mask - 1
                )
            else:
                decoder_position_ids = jnp.broadcast_to(
                    jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
                )

        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
        # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
        # it can be changed by FlaxWhisperAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(
            module,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
            **kwargs,
        ):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                position_ids=decoder_position_ids,
                **kwargs,
            )

        outputs = self.module.apply(
            inputs,
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            mutable=mutable,
            method=_decoder_forward,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past = outputs
            outputs["past_key_values"] = unfreeze(past["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past = outputs
            outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]

        return outputs

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_features: jnp.ndarray,
        decoder_input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        # prepare decoder inputs
        if decoder_position_ids is None:
            if decoder_attention_mask is not None:
                decoder_position_ids = (
                    decoder_attention_mask.cumsum(axis=-1) * decoder_attention_mask - 1
                )
            else:
                batch_size, sequence_length = decoder_input_ids.shape
                decoder_position_ids = jnp.broadcast_to(
                    jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
                )
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            input_features=jnp.array(input_features, dtype="f4"),
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
        )


@add_start_docstrings(
    "The bare Whisper Model transformer outputting raw hidden-states without any specific head on top.",
    WHISPER_START_DOCSTRING,
)
class FlaxWhisperModel(FlaxWhisperPreTrainedModel):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    module_class = FlaxWhisperModule


append_call_sample_docstring(
    FlaxWhisperModel,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxSeq2SeqModelOutput,
    _CONFIG_FOR_DOC,
)


class FlaxWhisperForConditionalGenerationModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.model = FlaxWhisperModule(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

    def _get_encoder_module(self):
        return self.model.encoder

    def _get_decoder_module(self):
        return self.model.decoder

    def __call__(
        self,
        input_features,
        decoder_input_ids,
        decoder_attention_mask: jnp.ndarray = None,
        decoder_position_ids: jnp.ndarray = None,
        position_ids: jnp.ndarray = None,
        attention_mask: jnp.ndarray = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        outputs = self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_embedding = self.model.decoder.embed_tokens.variables["params"][
                "embedding"
            ]
            lm_logits = self.lm_head.apply(
                {"params": {"kernel": shared_embedding.value.T}}, hidden_states
            )
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return output

        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


@add_start_docstrings(
    "The Whisper Model with a language modeling head.", WHISPER_START_DOCSTRING
)
class FlaxWhisperForConditionalGeneration(FlaxWhisperPreTrainedModel):
    module_class = FlaxWhisperForConditionalGenerationModule
    dtype: jnp.dtype = jnp.float32

    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=WhisperConfig
    )
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
        >>> input_features = inputs.input_features
        >>> encoder_outputs = model.encode(input_features=input_features)
        >>> decoder_start_token_id = model.config.decoder_start_token_id

        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> last_decoder_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        encoder_hidden_states = encoder_outputs[0]

        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `decoder_position_ids` when passing `past_key_values`."
                )

            if decoder_attention_mask is not None:
                decoder_position_ids = (
                    decoder_attention_mask.cumsum(axis=-1) * decoder_attention_mask - 1
                )
            else:
                decoder_position_ids = jnp.broadcast_to(
                    jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
                )
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length), dtype="i4")

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
        # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
        # it can be changed by FlaxWhisperAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(
            module,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
            **kwargs,
        ):
            decoder_module = module._get_decoder_module()
            outputs = decoder_module(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                position_ids=decoder_position_ids,
                **kwargs,
            )
            hidden_states = outputs[0]

            if self.config.tie_word_embeddings:
                shared_embedding = module.model.decoder.embed_tokens.variables[
                    "params"
                ]["embedding"]
                lm_logits = module.lm_head.apply(
                    {"params": {"kernel": shared_embedding.value.T}}, hidden_states
                )
            else:
                lm_logits = module.lm_head(hidden_states)

            return lm_logits, outputs

        outputs = self.module.apply(
            inputs,
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            mutable=mutable,
            method=_decoder_forward,
        )

        if past_key_values is None:
            lm_logits, decoder_outputs = outputs
        else:
            (lm_logits, decoder_outputs), past = outputs

        if return_dict:
            outputs = FlaxCausalLMOutputWithCrossAttentions(
                logits=lm_logits,
                hidden_states=decoder_outputs.hidden_states,
                attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
            )
        else:
            outputs = (lm_logits,) + decoder_outputs[1:]

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs["past_key_values"] = unfreeze(past["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]

        return outputs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jnp.DeviceArray] = None,
        decoder_attention_mask: Optional[jnp.DeviceArray] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # initializing the cache
        batch_size, seq_length = decoder_input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since the decoder uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, decoder_attention_mask, (0, 0)
            )
        else:
            position_ids = jnp.broadcast_to(
                jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
            )

        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
            "decoder_position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = (
            model_kwargs["decoder_position_ids"][:, -1:] + 1
        )
        return model_kwargs

    def generate(
        self,
        input_ids: jnp.ndarray,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        do_sample: Optional[bool] = None,
        prng_key: Optional[jnp.ndarray] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        num_beams: Optional[int] = None,
        no_repeat_ngram_size: Optional[int] = None,
        min_length: Optional[int] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        suppress_tokens: Optional[List[int]] = None,
        begin_suppress_tokens: Optional[List[int]] = None,
        forced_decoder_ids: Optional[List[List[int]]] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head. The method supports the following
        generation methods for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

            - *greedy decoding* by calling [`~generation.FlaxGenerationMixin._greedy_search`] if `num_beams=1` and
              `do_sample=False`.
            - *multinomial sampling* by calling [`~generation.FlaxGenerationMixin._sample`] if `num_beams=1` and
              `do_sample=True`.
            - *beam-search decoding* by calling [`~generation.FlaxGenerationMixin._beam_search`] if `num_beams>1` and
              `do_sample=False`.

        <Tip warning={true}>

        Apart from `inputs`, all the arguments below will default to the value of the attribute of the same name as
        defined in the model's config (`config.json`) which in turn defaults to the
        [`~modeling_utils.PretrainedConfig`] of the model.

        </Tip>

        Most of these parameters are explained in more detail in [this blog
        post](https://huggingface.co/blog/how-to-generate).

        Parameters:
            input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            max_length (`int`, *optional*, defaults to `model.config.max_length`):
                The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                `max_new_tokens`. In general, prefer the use of `max_new_tokens`, which ignores the number of tokens in
                the prompt.
            max_new_tokens (`int`, *optional*):
                The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            temperature (`float`, *optional*, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (`int`, *optional*, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher
                are kept for generation.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            bos_token_id (`int`, *optional*):
                The id of the *beginning-of-sequence* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            decoder_start_token_id (`int`, *optional*):
                If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
            suppress_tokens (`List[int]`, *optional*, defaults to model.config.suppress_tokens):
                A list of tokens that will be supressed at generation. The `FlaxSupressTokensLogitsProcessor` will set
                their log probs to `-inf` so that they are not sampled.
            begin_suppress_tokens (`List[int]`, *optional*, defaults to model.config.begin_supress_tokens):
                A list of tokens that will be supressed at the begining of the generation. The
                `FlaxSuppressTokensAtBeginLogitsProcessor` will set their log probs to `-inf` so that they are not
                sampled.
            forced_decoder_ids (`List[List[int]]`, *optional*, defaults to model.config.forced_decoder_ids):
                A list of pairs of integers which indicates a mapping from generation indices to token indices that
                will be forced before sampling. For example, `[[1, 123]]` means the second generated token will always
                be a token of index 123.
            trace (`bool`, *optional*, defaults to `True`):
                Whether to trace generation. Setting `trace=False` should only be used for debugging and will lead to a
                considerably slower runtime.
            params (`Dict[str, jnp.ndarray]`, *optional*):
                Optionally the model parameters can be passed. Can be useful for parallelized generation.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If the model
                is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs
                should be prefixed with *decoder_*. Also accepts `encoder_outputs` to skip encoder part.

        Return:
            [`~utils.ModelOutput`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

        >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        >>> model = FlaxAutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> input_context = "The dog"
        >>> # encode input context
        >>> input_ids = tokenizer(input_context, return_tensors="np").input_ids
        >>> # generate candidates using sampling
        >>> outputs = model.generate(input_ids=input_ids, max_length=20, top_k=30, do_sample=True)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ```"""
        # Validate the `.generate()` call
        self._validate_model_class()
        self._validate_model_kwargs(model_kwargs.copy())

        # set init values
        bos_token_id = (
            bos_token_id if bos_token_id is not None else self.config.bos_token_id
        )
        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.config.decoder_start_token_id
        )
        if forced_decoder_ids is None and hasattr(self.config, "forced_decoder_ids"):
            forced_decoder_ids = self.config.forced_decoder_ids
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        if decoder_start_token_id is None and self.config.is_encoder_decoder:
            raise ValueError(
                "`decoder_start_token_id` has to be defined for encoder-decoder generation."
            )

        # decoder-only models should use left-padding for generation (can't be checked with `trace=True`)
        if not self.config.is_encoder_decoder and not trace:
            if (
                pad_token_id is not None
                and jnp.sum(input_ids[:, -1] == pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        batch_size = input_ids.shape[0]

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            if model_kwargs.get("encoder_outputs") is None:
                model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                    input_ids, params, model_kwargs
                )
            # prepare decoder_input_ids for generation
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
            )

        # Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        if max_length is None and max_new_tokens is None:
            warnings.warn(
                "Neither `max_length` nor `max_new_tokens` have been set, `max_length` will default to "
                f"{self.config.max_length} (`self.config.max_length`). Controlling `max_length` via the config is "
                "deprecated and `max_length` will be removed from the config in v5 of Transformers -- we recommend "
                "using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids_seq_length
        elif max_length is not None and max_new_tokens is not None:
            raise ValueError(
                "Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- setting a"
                " limit to the generated output length. Remove one of those arguments. Please refer to the"
                " documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        # default to config if still None
        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length

        if min_length is not None and min_length > max_length:
            raise ValueError(
                f"Unfeasable length constraints: the minimum length ({min_length}) is larger than the maximum "
                f"length ({max_length})"
            )
        if input_ids_seq_length >= max_length:
            input_ids_string = (
                "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            )
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {max_length}. This can lead to unexpected behavior. You should consider increasing"
                "`max_new_tokens`."
            )

        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_beams = num_beams if num_beams is not None else self.config.num_beams

        if not do_sample and num_beams == 1:
            logits_processor = self._get_logits_processor(
                no_repeat_ngram_size,
                min_length,
                max_length,
                eos_token_id,
                forced_bos_token_id,
                forced_eos_token_id,
                input_ids_seq_length,
                suppress_tokens=suppress_tokens,
                begin_suppress_tokens=begin_suppress_tokens,
                forced_decoder_ids=forced_decoder_ids,
            )
            return self._greedy_search(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif do_sample and num_beams == 1:
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature
            )
            logits_processor = self._get_logits_processor(
                no_repeat_ngram_size,
                min_length,
                max_length,
                eos_token_id,
                forced_bos_token_id,
                forced_eos_token_id,
                input_ids_seq_length,
                suppress_tokens=suppress_tokens,
                begin_suppress_tokens=begin_suppress_tokens,
                forced_decoder_ids=forced_decoder_ids,
            )
            return self._sample(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                prng_key,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif not do_sample and num_beams > 1:
            # broadcast input_ids & encoder_outputs
            input_ids = self._expand_to_num_beams(input_ids, num_beams=num_beams)

            if "encoder_outputs" in model_kwargs:
                model_kwargs["encoder_outputs"][
                    "last_hidden_state"
                ] = self._expand_to_num_beams(
                    model_kwargs["encoder_outputs"]["last_hidden_state"],
                    num_beams=num_beams,
                )

            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = self._expand_to_num_beams(
                    model_kwargs["attention_mask"], num_beams=num_beams
                )

            logits_processor = self._get_logits_processor(
                no_repeat_ngram_size,
                min_length,
                max_length,
                eos_token_id,
                forced_bos_token_id,
                forced_eos_token_id,
                input_ids_seq_length,
                suppress_tokens=suppress_tokens,
                begin_suppress_tokens=begin_suppress_tokens,
                forced_decoder_ids=forced_decoder_ids,
            )

            return self._beam_search(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        else:
            raise NotImplementedError("`Beam sampling is currently not implemented.")

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            # Only use this arg if not None, otherwise just remove from model_kwargs
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
            if decoder_input_ids is not None:
                return decoder_input_ids
        decoder_start_token_id = self._get_decoder_start_token_id(
            decoder_start_token_id, bos_token_id
        )
        return (
            jnp.array(decoder_start_token_id, dtype="i4")
            .reshape(1, -1)
            .repeat(batch_size, axis=0)
        )

    def _get_decoder_start_token_id(
        self, decoder_start_token_id: int = None, bos_token_id: int = None
    ) -> int:
        # retrieve decoder_start_token_id for encoder-decoder models
        # fall back to bos_token_id if necessary
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.config.decoder_start_token_id
        )
        bos_token_id = (
            bos_token_id if bos_token_id is not None else self.config.bos_token_id
        )
        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "decoder_start_token_id")
            and self.config.decoder.decoder_start_token_id is not None
        ):
            return self.config.decoder.decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "bos_token_id")
            and self.config.decoder.bos_token_id is not None
        ):
            return self.config.decoder.bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    def _get_logits_processor(
        self,
        no_repeat_ngram_size: int,
        min_length: int,
        max_length: int,
        eos_token_id: int,
        forced_bos_token_id: int,
        forced_eos_token_id: int,
        input_ids_seq_length: int,
        suppress_tokens: Optional[List[int]] = None,
        begin_suppress_tokens: Optional[List[int]] = None,
        forced_decoder_ids: Optional[List[List[int]]] = None,
    ) -> FlaxLogitsProcessorList:
        """
        This class returns a [`FlaxLogitsProcessorList`] list object that contains all relevant [`FlaxLogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        processors = FlaxLogitsProcessorList()

        # init warp parameters
        no_repeat_ngram_size = (
            no_repeat_ngram_size
            if no_repeat_ngram_size is not None
            else self.config.no_repeat_ngram_size
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )
        forced_bos_token_id = (
            forced_bos_token_id
            if forced_bos_token_id is not None
            else self.config.forced_bos_token_id
        )
        forced_eos_token_id = (
            forced_eos_token_id
            if forced_eos_token_id is not None
            else self.config.forced_eos_token_id
        )
        suppress_tokens = (
            suppress_tokens
            if suppress_tokens is not None
            else self.config.suppress_tokens
        )
        begin_suppress_tokens = (
            begin_suppress_tokens
            if begin_suppress_tokens is not None
            else self.config.begin_suppress_tokens
        )

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(FlaxMinLengthLogitsProcessor(min_length, eos_token_id))
        if forced_bos_token_id is not None:
            processors.append(FlaxForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(
                FlaxForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id)
            )
        if suppress_tokens is not None:
            processors.append(FlaxSuppressTokensLogitsProcessor(suppress_tokens))
        if begin_suppress_tokens is not None:
            begin_index = input_ids_seq_length
            begin_index = (
                begin_index
                if (input_ids_seq_length > 1 or forced_bos_token_id is None)
                else begin_index + 1
            )
            if forced_decoder_ids is not None and len(forced_decoder_ids) > 0:
                begin_index += forced_decoder_ids[-1][
                    0
                ]  # generation starts after the last token that is forced
            processors.append(
                FlaxSuppressTokensAtBeginLogitsProcessor(
                    begin_suppress_tokens, begin_index
                )
            )
        if forced_decoder_ids is not None:
            forced_decoder_ids = [
                [input_ids_seq_length + i[0] - 1, i[1]] for i in forced_decoder_ids
            ]
            processors.append(FlaxForceTokensLogitsProcessor(forced_decoder_ids))
        return processors


FLAX_WHISPER_CONDITIONAL_GENERATION_DOCSTRING = r"""
    Returns:

    Transcription example:

    ```python
    >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
    >>> input_features = inputs.input_features
    >>> generated_ids = model.generate(input_ids=input_features)
    >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    >>> transcription
    ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
    ```
"""

overwrite_call_docstring(
    FlaxWhisperForConditionalGeneration,
    WHISPER_INPUTS_DOCSTRING + FLAX_WHISPER_CONDITIONAL_GENERATION_DOCSTRING,
)
append_replace_return_docstrings(
    FlaxWhisperForConditionalGeneration,
    output_type=FlaxSeq2SeqLMOutput,
    config_class=_CONFIG_FOR_DOC,
)


def load_pytorch_checkpoint_in_flax_state_dict(
    flax_model, pytorch_checkpoint_path, is_sharded, allow_missing_keys=False
):
    """Load pytorch checkpoints in a flax model"""
    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a PyTorch model in Flax, requires both PyTorch and Flax to be installed. Please see"
            " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation"
            " instructions."
        )
        raise

    if not is_sharded:
        pt_path = os.path.abspath(pytorch_checkpoint_path)
        logger.info(f"Loading PyTorch weights from {pt_path}")

        pt_state_dict = torch.load(pt_path, map_location="cpu")
        logger.info(
            f"PyTorch checkpoint contains {sum(t.numel() for t in pt_state_dict.values()):,} parameters."
        )

        flax_state_dict = convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model)
    else:
        # model is sharded and pytorch_checkpoint_path already contains the list of .pt shard files
        flax_state_dict = convert_pytorch_sharded_state_dict_to_flax(
            pytorch_checkpoint_path, flax_model
        )
    return flax_state_dict


def rename_key_and_reshape_tensor(
    pt_tuple_key: Tuple[str],
    pt_tensor: np.ndarray,
    random_flax_state_dict: Dict[str, jnp.ndarray],
    model_prefix: str,
) -> Tuple[Tuple[str], np.ndarray]:
    """Rename PT weight names to corresponding Flax weight names and reshape tensor if necessary"""

    def is_key_or_prefix_key_in_dict(key: Tuple[str]) -> bool:
        """Checks if `key` of `(prefix,) + key` is in random_flax_state_dict"""
        return len(set(random_flax_state_dict) & set([key, (model_prefix,) + key])) > 0

    # layer norm
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
    if pt_tuple_key[-1] in ["weight", "gamma"] and is_key_or_prefix_key_in_dict(
        renamed_pt_tuple_key
    ):
        return renamed_pt_tuple_key, pt_tensor

    # embedding
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("embedding",)
    if pt_tuple_key[-1] == "weight" and is_key_or_prefix_key_in_dict(
        renamed_pt_tuple_key
    ):
        return renamed_pt_tuple_key, pt_tensor

    # conv layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if (
        pt_tuple_key[-1] == "weight"
        and pt_tensor.ndim == 4
        and not is_key_or_prefix_key_in_dict(pt_tuple_key)
    ):
        pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
        return renamed_pt_tuple_key, pt_tensor

    # linear layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight" and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        pt_tensor = pt_tensor.T
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm weight
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight",)
    if pt_tuple_key[-1] == "gamma":
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm bias
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias",)
    if pt_tuple_key[-1] == "beta":
        return renamed_pt_tuple_key, pt_tensor

    return pt_tuple_key, pt_tensor


def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model):
    # convert pytorch tensor to numpy
    pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}

    model_prefix = flax_model.base_model_prefix
    random_flax_state_dict = flatten_dict(flax_model.params)
    flax_state_dict = {}

    load_model_with_head_into_base_model = (model_prefix not in flax_model.params) and (
        model_prefix in set([k.split(".")[0] for k in pt_state_dict.keys()])
    )
    load_base_model_into_model_with_head = (model_prefix in flax_model.params) and (
        model_prefix not in set([k.split(".")[0] for k in pt_state_dict.keys()])
    )

    # Need to change some parameters name to match Flax names
    for pt_key, pt_tensor in pt_state_dict.items():

        pt_tuple_key = tuple(pt_key.split("."))

        # remove base model prefix if necessary
        has_base_model_prefix = pt_tuple_key[0] == model_prefix
        if load_model_with_head_into_base_model and has_base_model_prefix:
            pt_tuple_key = pt_tuple_key[1:]

        # Correctly rename weight parameters
        flax_key, flax_tensor = rename_key_and_reshape_tensor(
            pt_tuple_key, pt_tensor, random_flax_state_dict, model_prefix
        )

        # add model prefix if necessary
        require_base_model_prefix = (model_prefix,) + flax_key in random_flax_state_dict
        if load_base_model_into_model_with_head and require_base_model_prefix:
            flax_key = (model_prefix,) + flax_key

        if flax_key in random_flax_state_dict:
            arr = random_flax_state_dict[flax_key]
            if hasattr(arr, "value"):
                arr = arr.value
            if flax_tensor.shape != arr.shape:
                raise ValueError(
                    f"PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                    f"{arr.shape}, but is {flax_tensor.shape}."
                )

        # also add unexpected weight so that warning is thrown
        if hasattr(random_flax_state_dict[flax_key], "value"):
            flax_state_dict[flax_key] = nn.LogicallyPartitioned(
                value=jnp.asarray(flax_tensor),
                names=random_flax_state_dict[flax_key].names,
            )
        else:
            flax_state_dict[flax_key] = jnp.asarray(flax_tensor)

    return unflatten_dict(flax_state_dict)


def convert_pytorch_sharded_state_dict_to_flax(shard_filenames, flax_model):
    import torch

    # Load the index
    flax_state_dict = {}
    for shard_file in shard_filenames:
        # load using msgpack utils
        pt_state_dict = torch.load(shard_file)
        pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}

        model_prefix = flax_model.base_model_prefix
        random_flax_state_dict = flatten_dict(flax_model.params)

        load_model_with_head_into_base_model = (
            model_prefix not in flax_model.params
        ) and (model_prefix in set([k.split(".")[0] for k in pt_state_dict.keys()]))
        load_base_model_into_model_with_head = (model_prefix in flax_model.params) and (
            model_prefix not in set([k.split(".")[0] for k in pt_state_dict.keys()])
        )
        # Need to change some parameters name to match Flax names
        for pt_key, pt_tensor in pt_state_dict.items():

            pt_tuple_key = tuple(pt_key.split("."))

            # remove base model prefix if necessary
            has_base_model_prefix = pt_tuple_key[0] == model_prefix
            if load_model_with_head_into_base_model and has_base_model_prefix:
                pt_tuple_key = pt_tuple_key[1:]

            # Correctly rename weight parameters
            flax_key, flax_tensor = rename_key_and_reshape_tensor(
                pt_tuple_key, pt_tensor, random_flax_state_dict, model_prefix
            )
            # add model prefix if necessary
            require_base_model_prefix = (
                model_prefix,
            ) + flax_key in random_flax_state_dict
            if load_base_model_into_model_with_head and require_base_model_prefix:
                flax_key = (model_prefix,) + flax_key

            if flax_key in random_flax_state_dict:
                if flax_tensor.shape != random_flax_state_dict[flax_key].shape:
                    raise ValueError(
                        f"PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                        f"{random_flax_state_dict[flax_key].shape}, but is {flax_tensor.shape}."
                    )

            # also add unexpected weight so that warning is thrown
            flax_state_dict[flax_key] = jnp.asarray(flax_tensor)
    return unflatten_dict(flax_state_dict)
