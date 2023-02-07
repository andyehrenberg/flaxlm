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

with open("requirements.txt", "r") as f:
    requirements = [line for line in f.readlines()]

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
    scripts=[],
    install_requires=requirements,
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
        'gpu': [f'jax[cuda] >= {_jax_version}'],
    },
)