import json
from absl import app, flags

import jax
import ml_collections
import transformers
import datasets
import jax.numpy as jnp
import jax.random as jrandom

import src.trainer as flax_trainer
import src.utils as utils
import src.data as data
import src.mesh_utils as mesh_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "json_config_path",
    None,
    "Path to the JSON serialized config for this training run.",
)


def train(_):
    with open(FLAGS.json_config_path, "rt") as fd:
        config = ml_collections.ConfigDict(json.load(fd))

    num_epochs = config.trainer_args.sampling_args.num_epochs
    save = config.trainer_args.save
    save_dir = config.trainer_args.output_dir
    tokenizer_path = config.trainer_args.model_args.tokenizer_path
    model_cls = config.trainer_args.model_args.model_cls

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rng = jrandom.PRNGKey(config.trainer_args.seed)

    mesh = mesh_utils.setup_mesh_and_partitioning_rules(
        config.trainer_args.parallelism_args
    )

    train_dataset = datasets.load_dataset("oscar", "unshuffled_deduplicated_no", split="train")
    eval_dataset = datasets.load_dataset("oscar", "unshuffled_deduplicated_no", split="test")

    train_dataset = data.PerHostDataset(
        dataset=train_dataset,
        global_data_shape=config.data_args.train.global_data_shape,
        global_mesh=trainer.mesh,
        data_axes=config.data_args.data_axes,
        tokenizer=tokenizer,
        input_ids_columns_name=config.data_args.train.input_ids_column_name,
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
    )

    eval_dataset = data.PerHostDataset(
        dataset=eval_dataset,
        global_data_shape=config.data_args.eval.global_data_shape,
        global_mesh=trainer.mesh,
        data_axes=config.data_args.data_axes,
        tokenizer=tokenizer,
        input_ids_columns_name=config.data_args.eval.input_ids_column_name,
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
    )

    steps_per_epoch = train_dataset._global_min_length
    config.trainer_args.num_train_steps = steps_per_epoch * num_epochs

    trainer = flax_trainer.Trainer(model_cls, config.trainer_args, mesh)

    def run_eval(trainer, eval_data):
        pass

    num_steps = 0

    eval_metrics = run_eval(trainer, eval_dataset)

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
