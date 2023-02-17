import ml_collections as mlc

from flaxlm.src.transformers_patch import (
    FlaxGPTJForCausalLM,
    FlaxT5ForConditionalGeneration,
)


def get_config():
    config = mlc.ConfigDict()
    config.directory_root = mlc.FieldReference("")
    experiment = mlc.ConfigDict()
    experiment.name = "Tiny T5X"
    experiment.description = "Training language models with Flax+JAX"

    trainer_args = mlc.ConfigDict()
    trainer_args.seed = 32
    trainer_args.save = False
    trainer_args.output_dir = ""
    trainer_args.sampling_args = dict(
        num_epochs=1,
        per_device_batch_size=2,
        gradient_accumulation_steps=1,
    )
    trainer_args.optimizer_args = dict(
        lr_init=3e-4,
        warmup_steps=100,
        max_grad_norm=1.0,
        use_dropout=True,
        half_precision=False,
    )
    trainer_args.parallelism_args = dict(
        mp_num=2,
        activation_partitioning_dims=1,
        parameter_partitioning_dims=1,
    )
    trainer_args.model_args = dict(
        model_cls=FlaxT5ForConditionalGeneration,
        pretrained_model_name_or_path="google/flan-t5-base",
        tokenizer_path="google/flan-t5-base",
        from_pt=False,
        gradient_checkpointing=True,
    )
    trainer_args.eval_args = dict(
        per_device_eval_batch_size=2,
        max_generation_new_tokens=256,
    )
    config.trainer_args = trainer_args

    data_args = mlc.ConfigDict()

    data_args.num_workers = 16
    data_args.tokenize_batch_size = 32
    data_args.group_batch_size = 32
    data_args.mode = "seq2seq"
    data_args.block_size = 1024
    data_args.pad_right = True
    data_args.max_len = 1024
    data_args.trunc_end = True
    data_args.decoder_max_len = 1024
    data_args.decoder_trunc_end = True
    data_args.decoder_prefix_str = "summarize: "
    data_args.train = dict(
        input_ids_column_name="article",
        remove_columns=["id"],
        decoder_input_ids_column_name="highlights",
        dataset="cnn_dailymail",
        dataset_name="3.0.0",
        dataset_split="train",
    )
    data_args.eval = dict(
        input_ids_column_name="article",
        remove_columns=["id"],
        decoder_input_ids_column_name="highlights",
        dataset="cnn_dailymail",
        dataset_name="3.0.0",
        dataset_split="validation",
    )

    config.data_args = data_args

    config.logging_args = dict(
        wandb_entity="andyehrenberg",
        wandb_project="flaxlm",
        wandb_job_type="tpu",
    )

    return config
