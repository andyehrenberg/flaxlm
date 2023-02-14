from functools import partial
from typing import Callable, Dict, Tuple, Type

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import flaxlm.src.partitioning_utils as partitioning_utils
import flaxlm.src.utils as utils
from chex import Array, Scalar
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import pjit

import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

P = PartitionSpec


def encoder_decoder_loss_fn(apply_fn, params, batch, use_dropout, dropout_rng):
    model_outputs = apply_fn(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        decoder_input_ids=batch["decoder_input_ids"],
        decoder_attention_mask=batch["decoder_attention_mask"],
        params=params,
        train=use_dropout,
        dropout_rng=dropout_rng if use_dropout else None,
    )
    per_token_loss = (
        optax.softmax_cross_entropy_with_integer_labels(
            model_outputs.logits[:, :-1], batch["decoder_input_ids"][:, 1:]
        )
        * batch["decoder_attention_mask"][:, 1:]
    )
    weight = batch["decoder_attention_mask"][:, 1:].sum()

    loss = per_token_loss.sum() / weight

    return loss, weight


def decoder_only_loss_fn(apply_fn, params, batch, use_dropout, dropout_rng):
    if hasattr(batch, "decoder_input_ids"):
        input_ids = jnp.concatenate(
            (batch["input_ids"], batch["decoder_input_ids"]),
            axis=1,
        )
        attention_mask = jnp.concatenate(
            (batch["attention_mask"], batch["decoder_attention_mask"]),
            axis=1,
        )
        position_ids = jnp.maximum(jnp.cumsum(attention_mask, axis=1) - 1, 0).astype(
            jnp.int32
        )
        model_outputs = apply_fn(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            params=params,
            train=use_dropout,
            dropout_rng=dropout_rng if use_dropout else None,
        )
        per_token_loss = (
            optax.softmax_cross_entropy_with_integer_labels(
                model_outputs.logits[:, (batch["input_ids"].shape[1] - 1) : -1, :],
                batch["decoder_input_ids"],
            )
            * batch["decoder_attention_mask"]
        )
        weight = batch["decoder_attention_mask"].sum()
        loss = per_token_loss.sum() / weight
    else:
        model_outputs = apply_fn(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            params=params,
            train=use_dropout,
            dropout_rng=dropout_rng if use_dropout else None,
        )
        per_token_loss = (
            optax.softmax_cross_entropy_with_integer_labels(
                model_outputs.logits[:, :-1], batch["input_ids"][:, 1:]
            )
            * batch["attention_mask"][:, 1:]
        )
        weight = batch["attention_mask"][:, 1:].sum()
        loss = per_token_loss.sum() / weight

    return loss, weight


class Trainer:
    def __init__(self, model_cls: Type, args: Dict, mesh: Mesh, num_train_steps: int):
        self.gradient_accumulation_steps = (
            args.sampling_args.gradient_accumulation_steps
        )
        self.lr_init = args.optimizer_args.lr_init
        self.warmup_steps = args.optimizer_args.warmup_steps
        self.max_grad_norm = args.optimizer_args.max_grad_norm
        self.use_dropout = args.optimizer_args.use_dropout
        self.num_epochs = args.sampling_args.num_epochs
        pretrained_path = args.model_args.pretrained_model_name_or_path
        self.num_train_steps = num_train_steps
        self.half_precision = args.optimizer_args.half_precision
        self.mp_num = args.parallelism_args.mp_num
        from_pt = args.model_args.from_pt
        gradient_checkpointing = args.model_args.gradient_checkpointing

        self.max_generation_new_tokens = args.eval_args.max_generation_new_tokens

        self.platform = jax.local_devices()[0].platform

        if self.half_precision:
            if self.platform == "tpu":
                self.dtype = jnp.bfloat16
            else:
                self.dtype = jnp.float16
        else:
            self.dtype = jnp.float32

        self.mesh = mesh

        rng = jax.random.PRNGKey(args.seed)
        rng, dropout_rng = jax.random.split(rng)

        model, eval_model, params = utils.setup_model(
            model_cls,
            pretrained_path,
            self.mp_num,
            from_pt,
            self.dtype,
            gradient_checkpointing,
        )
        self.model_config = model.config

        self.setup_train_state(model, eval_model, params, dropout_rng)
        del params
        self.batch_spec = P("batch")
        self.grad_batch_spec = P(None, "batch")

        self.train = self.with_mesh(jax.jit(self.make_train_step()))
        self.generate = self.with_mesh(
            jax.jit(
                partial(
                    self.make_generate(),
                    max_new_tokens=self.max_generation_new_tokens,
                )
            )
        )
        self.eval = self.with_mesh(jax.jit(self.make_eval_step()))

    def with_mesh(self, f):
        def wrapper(*args, **kwargs):
            with self.mesh:
                return f(*args, **kwargs)

        return wrapper

    def setup_train_state(
        self,
        model: Callable,
        eval_model: Callable,
        params: FrozenDict,
        dropout_rng: Array,
    ) -> None:
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=self.lr_init,
            transition_steps=self.warmup_steps + 1,
        )
        decay_fn = optax.linear_schedule(
            init_value=self.lr_init,
            end_value=0.0,
            transition_steps=self.num_train_steps - self.warmup_steps,
        )
        last_boundary = self.warmup_steps
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[last_boundary],
        )

        tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adamw(learning_rate=schedule_fn),
        )

        params = partitioning_utils.shard_logically_partitioned_params(
            params, self.mesh
        )

        def create_fn(params):
            return utils.TrainState.create(
                apply_fn=model.__call__,
                eval_apply_fn=eval_model.__call__,
                generate_fn=eval_model.generate,
                params=params,
                tx=tx,
                dropout_rng=dropout_rng,
            )

        train_state_shape = jax.eval_shape(create_fn, params)
        self.train_state_spec = nn.get_partition_spec(train_state_shape)
        self.param_spec = self.train_state_spec.params
        self.mesh_train_state_spec = nn.logical_to_mesh(self.train_state_spec)

        @jax.jit
        def partitioned_create(params):
            params = nn.with_logical_constraint(params, self.param_spec)
            train_state = create_fn(params)
            train_state = jax.lax.with_sharding_constraint(
                train_state, self.mesh_train_state_spec
            )

            return train_state

        p_create_fn = pjit.pjit(
            create_fn,
            in_shardings=(nn.logical_to_mesh(self.param_spec),),
            out_shardings=self.mesh_train_state_spec,
        )

        with self.mesh:
            #self.train_state = partitioned_create(params)
            self.train_state = p_create_fn(params)
        print(self.train_state.step.addressable_shards)

    def make_train_step(self) -> Callable:
        def train_step(
            train_state: utils.TrainState, batch: Dict
        ) -> Tuple[utils.TrainState, Dict]:
            print("Compiling train step")
            train_state = jax.lax.with_sharding_constraint(
                train_state, self.mesh_train_state_spec
            )

            batch = nn.with_logical_constraint(
                batch,
                self.batch_spec
                if self.gradient_accumulation_steps == 1
                else self.grad_batch_spec,
            )

            def get_minibatch(batch, grad_idx):
                return jax.tree_util.tree_map(
                    lambda x: jax.lax.dynamic_index_in_dim(x, grad_idx, keepdims=False),
                    batch,
                )

            def compute_loss(
                params: FrozenDict, batch: Dict, dropout_rng: Array
            ) -> Scalar:
                params = nn.with_logical_constraint(params, self.param_spec)

                if self.model_config.is_encoder_decoder:
                    return encoder_decoder_loss_fn(
                        train_state.apply_fn,
                        params,
                        batch,
                        self.use_dropout,
                        dropout_rng,
                    )
                else:
                    return decoder_only_loss_fn(
                        train_state.apply_fn,
                        params,
                        batch,
                        self.use_dropout,
                        dropout_rng,
                    )

            grad_fn = jax.value_and_grad(compute_loss, has_aux=True)

            # inspired by https://github.com/borisdayma/dalle-mini/blob/main/tools/train/train.py
            def loss_and_grad(grad_idx, dropout_rng):
                minibatch = (
                    get_minibatch(batch, grad_idx) if grad_idx is not None else batch
                )

                minibatch = nn.with_logical_constraint(minibatch, self.batch_spec)

                dropout_rng, _ = jrandom.split(dropout_rng)

                (loss, weight), grads = grad_fn(
                    train_state.params, minibatch, dropout_rng
                )

                grads = nn.with_logical_constraint(grads, self.param_spec)

                return loss, grads, weight, dropout_rng

            if self.gradient_accumulation_steps == 1:
                loss, grads, weight, dropout_rng = loss_and_grad(
                    None, train_state.dropout_rng
                )
                loss, grads = jax.tree_util.tree_map(
                    lambda x: x * weight, (loss, grads)
                )
                grads = nn.with_logical_constraint(grads, self.param_spec)
            else:
                init_carry = (
                    0.0,
                    nn.with_logical_constraint(
                        jax.tree_util.tree_map(jnp.zeros_like, train_state.params),
                        self.param_spec,
                    ),
                    0.0,
                    train_state.dropout_rng,
                )

                # inspired by https://github.com/borisdayma/dalle-mini/blob/main/tools/train/train.py
                def cumul_minibatch_step(
                    grad_idx: Scalar, carry: Tuple[Scalar, FrozenDict, Scalar]
                ) -> Tuple[Scalar, FrozenDict, Scalar]:
                    loss, grads, weight, dropout_rng = carry
                    sub_loss, sub_grads, sub_weight, dropout_rng = loss_and_grad(
                        grad_idx
                    )
                    sub_loss, sub_grads = jax.tree_util.tree_map(
                        lambda x: x * sub_weight, (sub_loss, sub_grads)
                    )
                    sub_grads = nn.with_logical_constraint(sub_grads, self.param_spec)
                    loss, grads, weight = jax.tree_util.tree_map(
                        jnp.add,
                        (loss, grads, weight),
                        (sub_loss, sub_grads, sub_weight),
                    )
                    grads = nn.with_logical_constraint(grads, self.param_spec)
                    return loss, grads, weight, dropout_rng

                (
                    loss,
                    grads,
                    weight,
                    dropout_rng,
                ) = jax.lax.fori_loop(
                    0,
                    self.gradient_accumulation_steps,
                    cumul_minibatch_step,
                    init_carry,
                )
                grads = nn.with_logical_constraint(grads, self.param_spec)

            metrics = {"loss": loss}

            grads, metrics = jax.tree_util.tree_map(
                lambda x: x / weight, (grads, metrics)
            )
            grads = nn.with_logical_constraint(grads, self.param_spec)

            new_train_state = train_state.apply_gradients(grads=grads)

            new_train_state = jax.lax.with_sharding_constraint(
                new_train_state, self.mesh_train_state_spec
            )

            return new_train_state, metrics

        return train_step

    def make_generate(self) -> Callable:
        def generate(
            train_state: utils.TrainState,
            input_ids: Array,
            attention_mask: Array,
            **kwargs
        ):
            train_state = jax.lax.with_sharding_constraint(
                train_state, self.mesh_train_state_spec
            )

            input_ids, attention_mask = jax.tree_util.tree_map(
                lambda x: nn.with_logical_constraint(x, self.batch_spec),
                (input_ids, attention_mask),
            )

            sequences = train_state.generate_fn(
                input_ids,
                attention_mask=attention_mask,
                params=train_state.params,
                **kwargs,
            ).sequences

            sequences = nn.with_logical_constraint(sequences, self.batch_spec)

            return sequences

        return generate

    def run_train(self, batch: Dict) -> Dict:
        self.train_state, metrics = self.train(self.train_state, batch)

        return metrics

    def make_eval_step(self) -> Callable:
        def eval_step(train_state: utils.TrainState, batch: Dict) -> Dict:
            print("Compiling eval step")
            train_state = jax.lax.with_sharding_constraint(
                train_state, self.mesh_train_state_spec
            )

            batch = nn.with_logical_constraint(batch, self.batch_spec)

            def compute_loss(params: FrozenDict, batch: Dict) -> Scalar:
                params = nn.with_logical_constraint(params, self.param_spec)

                if self.model_config.is_encoder_decoder:
                    return encoder_decoder_loss_fn(
                        train_state.apply_fn,
                        params,
                        batch,
                        False,
                        None,
                    )
                else:
                    return decoder_only_loss_fn(
                        train_state.apply_fn,
                        params,
                        batch,
                        False,
                        None,
                    )

            loss, weight = compute_loss(train_state.params, batch)

            return {"loss": loss, "weight": weight}

        return eval_step

    def run_eval(self, dataloader):
        losses, weights = 0.0, 0.0
        for batch in dataloader:
            metrics = self.eval(self.train_state, batch)
            losses += metrics["loss"] * metrics["weight"]
            weights += metrics["weight"]

        return {"eval loss": losses / weights}
