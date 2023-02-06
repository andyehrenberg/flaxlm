from absl import app, flags
import json

import jax
import jax.numpy as jnp
import jax.random as jrandom
import ml_collections as mlc

import src.data as data
import src.mesh_utils as mesh_utils
import src.partitioning_utils as partitioning_utils
import src.trainer as flax_trainer
import src.utils as utils

import transformers

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "json_config_path",
    None,
    "Path to the JSON serialized config for this training run.",
)


def train(_):
    with open(FLAGS.json_config_path, "rt") as fd:
        config = mlc.ConfigDict(json.load(fd))

    num_epochs = config.trainer_args.sampling_args.num_epochs
    save = config.trainer_args.save
    save_dir = config.trainer_args.output_dir
    tokenizer_path = config.trainer_args.model_args.tokenizer_path
    model_cls = config.trainer_args.model_args.model_cls

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rng = jrandom.PRNGKey(config.trainer_args.seed)

    mesh = mesh_utils.setup_mesh_and_partitioning_rules(
        config.trainer_args.parallelism_args
    )

    _, batch_size, _, _ = partitioning_utils.convert_per_device_batch_size(
        config.trainer_args.sampling_args.per_device_batch_size,
        config.trainer_args.parallelism_args.mp_num,
        config.trainer_args.sampling_args.gradient_accumulation_steps,
    )

    _, eval_batch_size, _, _ = partitioning_utils.convert_per_device_batch_size(
        config.trainer_args.eval_args.per_device_eval_batch_size,
        config.trainer_args.parallelism_args.mp_num,
        1,
    )

    train_shapes, eval_shapes, data_axes = utils.get_global_shape_dtypes(
        batch_size, eval_batch_size, config.data_args
    )

    train_dataset = data.PerHostDataset(
        dataset=config.data_args.train.dataset,
        global_data_shape=train_shapes,
        global_mesh=mesh,
        data_axes=data_axes,
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
    )

    eval_dataset = data.PerHostDataset(
        dataset=config.data_args.eval.dataset,
        global_data_shape=eval_shapes,
        global_mesh=mesh,
        data_axes=data_axes,
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
    )

    steps_per_epoch = train_dataset._global_min_length
    config.trainer_args.num_train_steps = steps_per_epoch * num_epochs

    trainer = flax_trainer.Trainer(model_cls, config.trainer_args, mesh)

    def run_eval(trainer, eval_data):
        pass

    num_steps = 0

    eval_metrics = run_eval(trainer, eval_dataset)
    log_metrics(eval_metrics, num_steps)

    for epoch in range(num_epochs):
        key, rng = jrandom.split(rng)
        for batch in train_dataset.set_epoch(key):
            metrics = trainer.train_step(batch)
            num_steps += 1
            if jax.process_index() == 0:
                log_metrics(metrics, num_steps)
        eval_metrics = run_eval(trainer, eval_dataset)
        log_metrics(eval_metrics, num_steps)

    if jax.process_index() == 0:
        if save:
            params = jax.device_get(trainer.train_state.params)
            utils.save_params(params, save_dir)


if __name__ == "__main__":
    app.run(train)
