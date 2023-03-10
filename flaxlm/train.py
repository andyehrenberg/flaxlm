from absl import app, flags

import jax
import jax.random as jrandom
from ml_collections import config_flags
import transformers
from time import time
import flax.linen as nn

import flaxlm.src.data as data
import flaxlm.src.mesh_utils as mesh_utils
import flaxlm.src.partitioning_utils as partitioning_utils
import flaxlm.src.trainer as flax_trainer
import flaxlm.src.utils as utils

config_flags.DEFINE_config_file("config")


def train(_):
    config = flags.FLAGS.config
    num_epochs = config.trainer_args.sampling_args.num_epochs
    save = config.trainer_args.save
    save_dir = config.trainer_args.output_dir
    tokenizer_path = config.trainer_args.model_args.tokenizer_path
    model_cls = config.trainer_args.model_args.model_cls
    gradient_accumulation_steps = (
        config.trainer_args.sampling_args.gradient_accumulation_steps
    )

    if jax.process_index() == 0:
        utils.init_logging(config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rng = jrandom.PRNGKey(config.trainer_args.seed)

    mesh, param_rules, compute_rules = mesh_utils.setup_mesh_and_partitioning_rules(
        config.trainer_args.parallelism_args
    )
    nn.set_logical_axis_rules(param_rules)

    batch_size = partitioning_utils.convert_per_device_batch_size(
        config.trainer_args.sampling_args.per_device_batch_size,
        config.trainer_args.parallelism_args.mp_num,
        gradient_accumulation_steps,
    )

    eval_batch_size = partitioning_utils.convert_per_device_batch_size(
        config.trainer_args.eval_args.per_device_eval_batch_size,
        config.trainer_args.parallelism_args.mp_num,
        1,
    )

    train_shapes, eval_shapes, train_axes, eval_axes = utils.get_global_shape_dtypes(
        batch_size,
        eval_batch_size,
        config.data_args,
        gradient_accumulation_steps,
    )

    train_dataset = data.PerHostDataset(
        dataset=config.data_args.train.dataset,
        global_data_shape=train_shapes,
        global_mesh=mesh,
        data_axes=train_axes,
        tokenizer=tokenizer,
        input_ids_column_name=config.data_args.train.input_ids_column_name,
        remove_columns=config.data_args.train.remove_columns,
        num_workers=config.data_args.num_workers,
        tokenize_batch_size=config.data_args.tokenize_batch_size,
        group_batch_size=config.data_args.group_batch_size,
        mode=config.data_args.mode,
        decoder_input_ids_column_name=config.data_args.train.decoder_input_ids_column_name,
        block_size=config.data_args.block_size,
        pad_value=tokenizer.pad_token_id,
        pad_right=config.data_args.pad_right,
        max_len=config.data_args.max_len,
        trunc_end=config.data_args.trunc_end,
        decoder_max_len=config.data_args.decoder_max_len,
        decoder_trunc_end=config.data_args.decoder_trunc_end,
        dataset_name=config.data_args.train.dataset_name,
        dataset_split=config.data_args.train.dataset_split,
        decoder_prefix_str=config.data_args.decoder_prefix_str,
    )

    eval_dataset = data.PerHostDataset(
        dataset=config.data_args.eval.dataset,
        global_data_shape=eval_shapes,
        global_mesh=mesh,
        data_axes=eval_axes,
        tokenizer=tokenizer,
        input_ids_column_name=config.data_args.eval.input_ids_column_name,
        remove_columns=config.data_args.eval.remove_columns,
        num_workers=config.data_args.num_workers,
        tokenize_batch_size=config.data_args.tokenize_batch_size,
        group_batch_size=config.data_args.group_batch_size,
        mode=config.data_args.mode,
        decoder_input_ids_column_name=config.data_args.eval.decoder_input_ids_column_name,
        block_size=config.data_args.block_size,
        pad_value=tokenizer.pad_token_id,
        pad_right=config.data_args.pad_right,
        max_len=config.data_args.max_len,
        trunc_end=config.data_args.trunc_end,
        decoder_max_len=config.data_args.decoder_max_len,
        decoder_trunc_end=config.data_args.decoder_trunc_end,
        dataset_name=config.data_args.eval.dataset_name,
        dataset_split=config.data_args.eval.dataset_split,
        decoder_prefix_str=config.data_args.decoder_prefix_str,
    )

    steps_per_epoch = train_dataset._global_min_length
    num_train_steps = steps_per_epoch * num_epochs

    trainer = flax_trainer.Trainer(
        model_cls,
        config.trainer_args,
        mesh,
        num_train_steps,
        param_rules,
        compute_rules,
    )

    num_steps = 0

    # eval_metrics = trainer.run_eval(eval_dataset.set_epoch(None))
    # if jax.process_index() == 0:
    # utils.log_metrics(eval_metrics, num_steps)

    for epoch in range(num_epochs):
        key, rng = jrandom.split(rng)
        for batch in train_dataset.set_epoch(key):
            t = time()
            metrics = trainer.run_train(batch)
            t1 = time()
            num_steps += 1
            utils.log_metrics({**metrics, "step time": t1 - t}, num_steps)
        eval_metrics = trainer.run_eval(eval_dataset.set_epoch(None))
        if jax.process_index() == 0:
            utils.log_metrics(eval_metrics, num_steps)

    if jax.process_index() == 0:
        if save:
            params = jax.device_get(trainer.train_state.params)
            utils.save_params(params, save_dir)


if __name__ == "__main__":
    app.run(train)
