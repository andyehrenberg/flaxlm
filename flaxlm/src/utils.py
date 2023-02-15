import os
from typing import Callable

import flax
import flax.core.frozen_dict as frozen_dict
import flax.linen as nn
import flax.serialization as serialization
from flax.training import checkpoints, train_state
import orbax.checkpoint as orbax
import jax
import jax.numpy as jnp
import ml_collections as mlc
from jax.sharding import PartitionSpec
import wandb

P = PartitionSpec


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray
    eval_apply_fn: Callable = flax.struct.field(pytree_node=False)
    generate_fn: Callable = flax.struct.field(pytree_node=False)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=jnp.array(0),
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


def setup_model(
    model_cls,
    pretrained_path,
    mp_num,
    from_pt,
    dtype,
    gradient_checkpointing,
    randomize=False,
    config=None,
):
    with jax.default_device(jax.devices("cpu")[0]):
        if not randomize:
            model = model_cls.from_pretrained(
                pretrained_path, from_pt=from_pt, dtype=dtype
            )
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
                        model.params["model"]["decoder"]["embed_tokens"][
                            "embedding"
                        ].value
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
        else:
            if not config:
                model = model_cls.from_pretrained(pretrained_path)
                config = model.config
            original_vocab = config.vocab_size

            if gradient_checkpointing:
                config.gradient_checkpointing = True

            if mp_num > 1:
                remainder = original_vocab % mp_num
                config.vocab_size = original_vocab + mp_num - remainder

            model = model_cls(config, dtype=dtype)
            params = model.params

        eval_model = (
            model
            if dtype == jnp.float32
            else model_cls(config, _do_init=False, dtype=jnp.float32)
        )
        if config.vocab_size != original_vocab:
            model.config.suppress_tokens += list(
                range(original_vocab, config.vocab_size)
            )
            eval_model.config.suppress_tokens += list(
                range(original_vocab, config.vocab_size)
            )

    return model, eval_model, frozen_dict.freeze(params)


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


def save_checkpoint(train_state, ckpt_dir, step):
    if jax.process_count() > 1:
        async_checkpointer = orbax.AsyncCheckpointer(
            orbax.PyTreeCheckpointHandler(), timeout_secs=50
        )
        checkpoints.save_checkpoint_multiprocess(
            ckpt_dir,
            train_state,
            step=step,
            overwrite=True,
            keep=4,
            orbax_checkpointer=async_checkpointer,
        )
    else:
        orbax_checkpointer = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
        checkpoints.save_checkpoint(
            ckpt_dir,
            train_state,
            step=step,
            overwrite=True,
            keep=4,
            orbax_checkpointer=orbax_checkpointer,
        )


def restore_checkpoint(target, ckpt_dir, step=0):
    if jax.process_count() > 1:
        async_checkpointer = orbax.AsyncCheckpointer(
            orbax.PyTreeCheckpointHandler(), timeout_secs=50
        )
        restored = checkpoints.restore_checkpoint(
            ckpt_dir,
            target=target,
            step=step,
            orbax_checkpointer=async_checkpointer,
        )
    else:
        restored = checkpoints.restore_checkpoint(ckpt_dir, target=target, step=step)

    return restored


def get_global_shape_dtypes(
    train_batch_size, eval_batch_size, data_args, gradient_accumulation_steps=1
):
    if data_args.mode == "clm":
        block_size = data_args.block_size
        eval_dtype_struct = jax.ShapeDtypeStruct((eval_batch_size, block_size), "i4")
        if gradient_accumulation_steps == 1:
            train_dtype_struct = jax.ShapeDtypeStruct(
                (train_batch_size, block_size), "i4"
            )
        else:
            per_grad_step = train_batch_size // gradient_accumulation_steps
            train_dtype_struct = jax.ShapeDtypeStruct(
                (gradient_accumulation_steps, per_grad_step, block_size),
                "i4",
            )
        train_global_data_shape = {
            "input_ids": train_dtype_struct,
            "attention_mask": train_dtype_struct,
        }
        eval_global_data_shape = {
            "input_ids": eval_dtype_struct,
            "attention_mask": eval_dtype_struct,
        }
        eval_axes = {
            "input_ids": P("batch"),
            "attention_mask": P("batch"),
        }
        if gradient_accumulation_steps == 1:
            train_axes = {
                "input_ids": P("batch"),
                "attention_mask": P("batch"),
            }
        else:
            train_axes = {
                "input_ids": P(None, "batch"),
                "attention_mask": P(None, "batch"),
            }
    elif data_args.mode == "seq2seq":
        input_ids_len = data_args.max_len
        if data_args.train.decoder_input_ids_column_name:
            decoder_ids_len = data_args.decoder_max_len
            eval_encoder_dtype_struct = jax.ShapeDtypeStruct(
                (eval_batch_size, input_ids_len), "i4"
            )
            eval_decoder_dtype_struct = jax.ShapeDtypeStruct(
                (eval_batch_size, decoder_ids_len), "i4"
            )
            if gradient_accumulation_steps == 1:
                train_encoder_dtype_struct = jax.ShapeDtypeStruct(
                    (train_batch_size, input_ids_len), "i4"
                )
                train_decoder_dtype_struct = jax.ShapeDtypeStruct(
                    (train_batch_size, decoder_ids_len), "i4"
                )
            else:
                per_grad_step = train_batch_size // gradient_accumulation_steps
                train_encoder_dtype_struct = jax.ShapeDtypeStruct(
                    (gradient_accumulation_steps, train_batch_size, input_ids_len), "i4"
                )
                train_decoder_dtype_struct = jax.ShapeDtypeStruct(
                    (gradient_accumulation_steps, train_batch_size, decoder_ids_len),
                    "i4",
                )
            train_global_data_shape = {
                "input_ids": train_encoder_dtype_struct,
                "attention_mask": train_encoder_dtype_struct,
                "decoder_input_ids": train_decoder_dtype_struct,
                "decoder_attention_mask": train_decoder_dtype_struct,
            }
            eval_global_data_shape = {
                "input_ids": eval_encoder_dtype_struct,
                "attention_mask": eval_encoder_dtype_struct,
                "decoder_input_ids": eval_decoder_dtype_struct,
                "decoder_attention_mask": eval_decoder_dtype_struct,
            }
            eval_axes = {
                "input_ids": P("batch"),
                "attention_mask": P("batch"),
                "decoder_input_ids": P("batch"),
                "decoder_attention_mask": P("batch"),
            }
            if gradient_accumulation_steps == 1:
                train_axes = {
                    "input_ids": P("batch"),
                    "attention_mask": P("batch"),
                    "decoder_input_ids": P("batch"),
                    "decoder_attention_mask": P("batch"),
                }
            else:
                train_axes = {
                    "input_ids": P(None, "batch"),
                    "attention_mask": P(None, "batch"),
                    "decoder_input_ids": P(None, "batch"),
                    "decoder_attention_mask": P(None, "batch"),
                }
        else:
            eval_dtype_struct = jax.ShapeDtypeStruct(
                (eval_batch_size, input_ids_len), "i4"
            )
            if gradient_accumulation_steps == 1:
                train_dtype_struct = jax.ShapeDtypeStruct(
                    (train_batch_size, input_ids_len), "i4"
                )
            else:
                per_grad_step = train_batch_size // gradient_accumulation_steps
                train_dtype_struct = jax.ShapeDtypeStruct(
                    (gradient_accumulation_steps, train_batch_size, input_ids_len), "i4"
                )
            train_global_data_shape = {
                "input_ids": train_dtype_struct,
                "attention_mask": train_dtype_struct,
            }
            eval_global_data_shape = {
                "input_ids": eval_dtype_struct,
                "attention_mask": eval_dtype_struct,
            }
            if gradient_accumulation_steps == 1:
                train_axes = {
                    "input_ids": P("batch"),
                    "attention_mask": P("batch"),
                }
            else:
                train_axes = {
                    "input_ids": P(None, "batch"),
                    "attention_mask": P(None, "batch"),
                }
    else:
        raise NotImplementedError("Mode can either be seq2seq or clm")

    return train_global_data_shape, eval_global_data_shape, train_axes, eval_axes


def init_logging(config):
    if jax.process_index() == 0:
        wandb.init(
            entity=config.logging_args.wandb_entity,
            project=config.logging_args.wandb_project,
            job_type=config.logging_args.wandb_job_type,
            config=config,
        )


def log_metrics(metrics, step):
    if jax.process_index() == 0:
        wandb.log(metrics, step=step)
