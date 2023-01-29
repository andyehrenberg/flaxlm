import json

import jax
import ml_collections
import transformers
from absl import app, flags

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

    train_data = data.prepare_data(config.data_args.train.dataset, tokenizer)
    eval_data = data.prepare_data(config.data_args.eval.dataset, tokenizer)


    trainer = flax_trainer.Trainer(model_cls, config.trainer_args)

    def run_eval(trainer, eval_data):
        pass

    num_steps = 0

    eval_metrics = run_eval(trainer, eval_data)

    for epoch in range(num_epochs):
        loader = data.get_per_replica_data_pipeline(
            train_data,
            pytree_of_ShapedTypeStructs,
            trainer.mesh,
            pytree_of_partition_specs,
        )
            # trainer.batch_size_per_host
        for batch in loader:
            metrics = trainer.train_step(batch)
        eval_metrics = run_eval(trainer, eval_data)

    if jax.process_index() == 0:
        if save:
            params = jax.device_get(trainer.train_state.params)
            utils.save_params(params, save_dir)


if __name__ == "__main__":
    app.run(train)
