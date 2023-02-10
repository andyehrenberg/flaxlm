import os
import sys
import setuptools

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'flaxlm')
sys.path.append(version_path)
from version import __version__ 

# Get the long description from the README file.
with open('README.md') as fp:
    long_description = fp.read()

_jax_version = '0.4.2'

with open("requirements.txt", "r") as f:
    requirements = [line for line in f.readlines() if line[0] != "#"]

print(setuptools.find_packages())

setuptools.setup(
    name='flaxlm',
    version=__version__,
    description='Flexible parallelism for training language models with Flax/JAX',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Andy Ehrenberg',
    author_email='andyehrenberg@gmail.com',
    url='http://github.com/andyehrenberg/flaxlm',
    packages=setuptools.find_packages(),
    scripts=[],
    install_requires=requirements,
    extras_require={
        'gcp': [
            'gevent',
            'google-api-python-client',
            'google-compute-engine',
            'google-cloud-storage==2.5.0',
            'oauth2client',
        ],
        # Cloud TPU requirements.
        'tpu': [f'jax[tpu] >= {_jax_version}'],
        'gpu': [f'jax[cuda] >= {_jax_version}'],
        'cpu': [f'jax[cpu] >= {_jax_version}'],
    },
)
