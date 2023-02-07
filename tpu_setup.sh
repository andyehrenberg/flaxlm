git clone https://github.com/andyehrenberg/tiny_t5x.git
cd tiny_t5x
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
wandb login