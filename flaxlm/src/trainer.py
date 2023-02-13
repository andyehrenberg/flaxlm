from functools import partial
from typing import Callable, Dict, Tuple, Type

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.experimental.pjit as pjit
import optax
import src.partitioning_utils as partitioning_utils
import src.utils as utils
from chex import Array, Scalar
from jax.sharding import Mesh, PartitionSpec

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

        self.per_device_batch_size = args.sampling_args.per_device_batch_size
        self.per_device_eval_batch_size = args.eval_args.per_device_eval_batch_size
        self.max_generation_new_tokens = args.eval_args.max_generation_new_tokens

        (
            self.per_node_per_grad_step_batch_size,
            self.batch_size,
            self.loader_batch_size,
            self.node_groups,
        ) = partitioning_utils.convert_per_device_batch_size(
            self.per_device_batch_size, self.mp_num, self.gradient_accumulation_steps
        )

        (
            self.per_node_eval_batch_size,
            self.eval_batch_size,
            _,
            _,
        ) = partitioning_utils.convert_per_device_batch_size(
            self.per_device_eval_batch_size, self.mp_num, 1
        )
        self.eval_loader_batch_size = self.eval_batch_size * self.node_groups

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

        if self.half_precision and self.platform != "tpu":
            dynamic_scale = utils.DynamicScale()
        else:
            dynamic_scale = utils.NoOp()

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
                dynamic_scale=dynamic_scale,
                dropout_rng=dropout_rng,
            )

        train_state_shape = jax.eval_shape(create_fn, params)
        self.train_state_spec = nn.get_partition_spec(train_state_shape)
        print(self.train_state_spec.step)
        print(self.train_state_spec.tx)
        print(self.train_state_spec.dynamic_scale)
        train_state_spec = nn.logical_to_mesh(self.train_state_spec)
        print(nn.get_logical_axis_rules())

        @self.with_mesh
        @jax.jit
        def partitioned_create(params):
            train_state = create_fn(params)
            train_state = jax.lax.with_sharding_constraint(
                train_state, train_state_spec
            )
            #train_state = nn.with_logical_constraint(train_state, self.train_state_spec)

            return train_state

        p_create_fn = self.with_mesh(
            pjit.pjit(
                create_fn,
                in_axis_resources=train_state_spec.params,
                out_axis_resources=train_state_spec,
            )
        )

        # self.train_state = p_create_fn(params)
        self.train_state = partitioned_create(params)

        self.param_spec = self.train_state_spec.params

    def make_train_step(self) -> Callable:
        def train_step(
            train_state: utils.TrainState, batch: Dict
        ) -> Tuple[utils.TrainState, Dict]:
            # train_state = nn.with_logical_constraint(train_state, self.train_state_spec)

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

            dynamic_scale = train_state.dynamic_scale

            grad_fn = dynamic_scale.value_and_grad(compute_loss, has_aux=True)

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
                dynamic_scale, finite = dynamic_scale.update(grads)
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
                    dynamic_scale,
                    jnp.array(True),
                    train_state.dropout_rng,
                )

                # inspired by https://github.com/borisdayma/dalle-mini/blob/main/tools/train/train.py
                def cumul_minibatch_step(
                    grad_idx: Scalar, carry: Tuple[Scalar, FrozenDict, Scalar]
                ) -> Tuple[Scalar, FrozenDict, Scalar]:
                    loss, grads, weight, dynamic_scale, finite, dropout_rng = carry
                    sub_loss, sub_grads, sub_weight, dropout_rng = loss_and_grad(
                        grad_idx
                    )
                    dynamic_scale, sub_finite = dynamic_scale.update(sub_grads)
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
                    finite = finite & sub_finite
                    return loss, grads, weight, dynamic_scale, finite, dropout_rng

                (
                    loss,
                    grads,
                    weight,
                    dynamic_scale,
                    finite,
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

            # new_train_state = nn.with_logical_constraint(
            #    new_train_state, self.train_state_spec
            # )

            if self.half_precision:
                new_train_state = new_train_state.replace(
                    opt_state=jax.tree_util.tree_map(
                        partial(jnp.where, finite),
                        new_train_state.opt_state,
                        train_state.opt_state,
                    ),
                    params=jax.tree_util.tree_map(
                        partial(jnp.where, finite),
                        new_train_state.params,
                        train_state.params,
                    ),
                    dynamic_scale=dynamic_scale,
                    dropout_rng=dropout_rng,
                )

            # new_train_state = nn.with_logical_constraint(
            #    new_train_state, self.train_state_spec
            # )

            return new_train_state, metrics

        return train_step

    def make_generate(self) -> Callable:
        def generate(
            train_state: utils.TrainState,
            input_ids: Array,
            attention_mask: Array,
            **kwargs
        ):
            train_state = nn.with_logical_constraint(train_state, self.train_state_spec)

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

    def train_step(self, batch: Dict) -> Dict:
        bs_shape = (self.per_node_per_grad_step_batch_size * self.node_groups,)
        if self.gradient_accumulation_steps > 1:
            # reshape data into (gradient_accumulation_steps, batch_per_node, ...)
            # to avoid any data redistribution when sharding
            bs_shape = (self.gradient_accumulation_steps,) + bs_shape

        batch = jax.tree_util.tree_map(
            lambda x: x.reshape(bs_shape + x.shape[1:]),
            batch,
        )

        self.train_state, metrics = self.train(self.train_state, batch)

        return metrics
