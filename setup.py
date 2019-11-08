# Written by Curtis G. Northcutt

# For pypi upload
# 0. python setup.py check -r -s
# 0. rm -rf dist build
# 1. python setup.py sdist bdist_wheel --universal
# 2. python3 -m twine upload dist/*

from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# Get version number
exec(open('cleanlab/version.py').read())

setup(
    name='cleanlab',
    version=__version__,
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    description = 'The Python package for cleaning and learning with noisy labels. Works for all noisy label distributions, datasets, and models.',
    url = 'https://github.com/cgnorthcutt/cleanlab',
    author = 'Curtis G. Northcutt',
    author_email = 'cgn@mit.edu',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
      'Development Status :: 3 - Alpha',

      'Intended Audience :: Developers',
      'Intended Audience :: Education',
      'Intended Audience :: Science/Research',

      'License :: OSI Approved :: MIT License',

      # We believe this package works will all versions, but we do not guarantee it!
      'Programming Language :: Python :: 2.7',
      'Programming Language :: Python :: 3.4',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',

      'Programming Language :: Python',
      'Topic :: Software Development',
      'Topic :: Scientific/Engineering',
      'Topic :: Scientific/Engineering',
      'Topic :: Scientific/Engineering :: Mathematics',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development',
      'Topic :: Software Development :: Libraries',
      'Topic :: Software Development :: Libraries :: Python Modules',

      'Operating System :: Microsoft :: Windows',
      'Operating System :: POSIX',
      'Operating System :: Unix',
      'Operating System :: MacOS',
    ],

    # What does your project relate to?
    keywords='machine_learning denoising classification weak_supervision learning_with_noisy_labels unsupervised_learning',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['img', 'examples']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy>=1.11.3', 'scikit-learn>=0.18', 'scipy>=1.1.0', ],
)
