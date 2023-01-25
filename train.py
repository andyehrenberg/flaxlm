import json
from time import time

import jax
import ml_collections
import transformers
from absl import app, flags

import trainer as trainer
import utils

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
    context_size = config.trainer_args.sampling_args.max_prompt_length
    decoder_max_positions = (
        config.trainer_args.sampling_args.max_target_length + context_size
    )
    assert (
        decoder_max_positions <= 448
    ), "decoder_max_positions can't exceed Whisper's context size"

    num_train_steps = config.trainer_args.sampling_args.num_train_steps
    eval_interval = config.trainer_args.eval_args.eval_interval
    tokenizer_path = config.trainer_args.model_args.tokenizer_path
    model_cls = config.trainer_args.model_args.model_cls
    num_epochs = num_train_steps // eval_interval

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    trainer = trainer.Trainer(model_cls, config.trainer_args)

    def run_eval(trainer, eval_data):
        pass

    num_steps = 0

    eval_metrics = run_eval(trainer, eval_data)

    for epoch in range(num_epochs):
        loader = make_loader(trainer.batch_size_per_host)
        for batch in loader:
            metrics = trainer.train_step(batch)
        eval_metrics = run_eval(trainer, eval_data)

    if jax.process_index() == 0:
        if save:
            params = jax.device_get(trainer.train_state.params)
            utils.save_params(params, save_dir)


if __name__ == "__main__":
    app.run(train)
