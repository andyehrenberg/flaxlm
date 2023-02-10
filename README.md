# flaxlm

Training/fine-tuning language models with flexible, user-defined data and model parallelism using new parallelism features from Flax and JAX.

Uses `nn.with_logical_partitioning` to give logical axis metadata to model parameters, and `nn.with_logical_constraint` to provide guidance to the compiler on how to shard data/parameters/optimizer states/gradients. Leverages the new parallelism capabilities of `jax.jit`.

Supports pre-training/finetuning of GPT-J and OPT, and finetuning of T5.

Also gives a demonstration of multi-host dataloading that can handle scenarios where model replicas span across multiple hosts.

Takes inspiration from and uses some of the great utilities found in:

https://github.com/Sea-Snell/JAXSeq (gradient checkpointing, working with huggingface models)

https://github.com/sholtodouglas/multihost_dataloading (multi-host dataloading strategies)

https://github.com/google-research/t5x (construction of meshes for different platforms)

https://github.com/borisdayma/dalle-mini (determining per-host and global batch sizes, train steps that efficiently support gradient accumulation, mp and dp)

https://github.com/stanford-crfm/levanter (multihost utils, getting layout of devices)

Running on TPU:

```bash
git clone https://github.com/andyehrenberg/flaxlm.git
cd flaxlm
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -m wandb login

python3 flaxlm/train.py --config=flaxlm/src/config/config.py
```