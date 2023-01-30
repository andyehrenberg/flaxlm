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
    num_train_steps = config.trainer_args.sampling_args.num_train_steps
    eval_interval = config.trainer_args.eval_args.eval_interval
    tokenizer_path = config.trainer_args.model_args.tokenizer_path
    model_cls = config.trainer_args.model_args.model_cls
    num_epochs = num_train_steps // eval_interval

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rng = jrandom.PRNGKey(config.trainer_args.seed)

    trainer = flax_trainer.Trainer(model_cls, config.trainer_args)

    train_dataset = datasets.load_dataset("oscar", "unshuffled_deduplicated_no", split="train")

    train_dataset = data.PerHostDataset(
        train_dataset,
        config.data_args.global_data_shape,
        trainer.mesh,
        config.data_args.data_axes,
        tokenizer,
        config.data_args.text_column_name,
        config.data_args.remove_columns,
        config.data_args.num_workers,
        config.data_args.block_size,
        config.data_args.tokenize_batch_size,
        config.data_args.group_batch_size,
    )

    #train_data = data.prepare_data(config.data_args.train.dataset, tokenizer)
    #eval_data = data.prepare_data(config.data_args.eval.dataset, tokenizer)

    def run_eval(trainer, eval_data):
        pass

    num_steps = 0

    eval_metrics = run_eval(trainer, eval_data)

    for epoch in range(num_epochs):
        key, rng = jrandom.split(rng)
        for batch in train_dataset.set_epoch(key):
            metrics = trainer.train_step(batch)
            num_steps += 1
            if jax.process_index() == 0:
                log_metrics(metrics, num_steps)
        eval_metrics = run_eval(trainer, eval_data)
        log_metrics(eval_metrics, num_steps)

    if jax.process_index() == 0:
        if save:
            params = jax.device_get(trainer.train_state.params)
            utils.save_params(params, save_dir)


if __name__ == "__main__":
    app.run(train)
