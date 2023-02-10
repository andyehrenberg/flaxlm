git clone https://github.com/andyehrenberg/flaxlm.git
cd flaxlm
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -m wandb login