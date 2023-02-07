import os
import sys
import setuptools

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'tiny_t5x')
sys.path.append(version_path)

from version import __version__  # pylint: disable=g-import-not-at-top

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

_jax_version = '0.4.3'
_jaxlib_version = '0.4.2'

setuptools.setup(
    name='tiny_t5x',
    version=__version__,
    description='Flexible parallelism for training transformers',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Andy Ehrenberg',
    author_email='andyehrenberg@gmail.com',
    url='http://github.com/andyehrenberg/tiny_t5x',
    packages=setuptools.find_packages(),
    package_data={
        #'': ['**/*.gin'],  # not all subdirectories may have __init__.py.
    },
    scripts=[],
    install_requires=[
        'absl-py',
        f"jax >= {_jax_version}",
        f"jaxlib >= {_jaxlib_version}",
        "orbax @ git+https://github.com/google/orbax#egg=orbax",
        "tensorflow-cpu",
        "tensorstore >= 0.1.20",
        "chex==0.1.6",
        "datasets==2.8.0",
        "flax==0.6.4",
        "google-cloud-storage==2.5.0",
        "ml_collections==0.1.1",
        "numpy==1.24.1",
        "optax==0.1.4",
        "transformers==4.26.0",
        "tqdm==4.64.0",
        "wandb==0.13.9",
    ],
    extras_require={
        'gcp': [
            'gevent',
            'google-api-python-client',
            'google-compute-engine',
            'google-cloud-storage',
            'oauth2client',
        ],
        # Cloud TPU requirements.
        'tpu': [f'jax[tpu] >= {_jax_version}'],
        'gpu': [],
    },
)