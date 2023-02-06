import ml_collections as mlc

from ..transformers_patch import FlaxGPTJForCausalLM, FlaxT5ForConditionalGeneration


class Config(mlc.ConfigDict):
    def __init__(self):
        super().__init__()
        # Define base attributes common to all experiments.
        self.directory_root = mlc.FieldReference("")
        experiment = mlc.ConfigDict()
        experiment.name = "Tiny T5X"
        experiment.description = "Training language models with Flax+JAX"
        experiment.config = self.get_ref("directory_root") + "configs/config.json"
        experiment.env_args = {
            "COMET_API_KEY": "",
        }
        self.experiment = experiment

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
            model_cls=FlaxGPTJForCausalLM,
            pretrained_model_name_or_path="EleutherAI/gpt-j-6b",
            tokenizer_path="EleutherAI/gpt-j-6b",
            from_pt=False,
            gradient_checkpointing=True,
        )
        trainer_args.eval_args = dict(
            per_device_eval_batch_size=2,
            max_generation_new_tokens=256,
        )
        self.trainer_args = trainer_args

        data_args = mlc.ConfigDict()

        data_args.num_workers = 16
        data_args.tokenize_batch_size = 32
        data_args.group_batch_size = 32
        data_args.mode = "clm"
        data_args.block_size = 1024
        data_args.pad_right = True
        data_args.max_len = 1024
        data_args.trunc_end = True
        data_args.decoder_max_len = 1024
        data_args.decoder_trunc_end = True
        data_args.decoder_prefix_str = "summarize: "
        data_args.train = dict(
            input_ids_column_name="text",
            remove_columns=["id", "text"],
            decoder_input_ids_column_name=None,
            dataset="oscar",
            dataset_name="unshuffled_deduplicated_no",
            dataset_split="train",
        )
        data_args.eval = dict(
            input_ids_column_name="text",
            remove_columns=["id", "text"],
            decoder_input_ids_column_name=None,
            dataset="oscar",
            dataset_name="unshuffled_deduplicated_no",
            dataset_split="test",
        )

        self.data_args = data_args
