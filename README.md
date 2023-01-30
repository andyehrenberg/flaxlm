# tiny_t5x

Training language models with flexible, user-defined data and model parallelism using new parallelism features from Flax and JAX.

Uses `nn.with_logical_partitioning` to give logical axis metadata to model parameters, and `nn.with_logical_constraint` to provide guidance to the compiler on how to shard activations/losses/gradients.
