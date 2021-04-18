# Written by Curtis G. Northcutt

# For pypi upload
# 0. python setup.py check -r -s
# 0. rm -rf dist build
# 1. python setup.py sdist bdist_wheel --universal
# 1. twine check dist/*
# 2. python3 -m twine upload dist/*

from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info

# To use a consistent encoding
from codecs import open
from os import path


class egg_info_ex(egg_info):
    """Includes license file into `.egg-info` folder."""

    def run(self):
        # don't duplicate license into `.egg-info` when building a distribution
        if not self.distribution.have_run.get('install', True):
            # `install` command is in progress, copy license
            self.mkpath(self.egg_info)
            self.copy_file('LICENSE', self.egg_info)

        egg_info.run(self)


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# Get version number
exec(open('cleanlab/version.py').read())



setup(
    name='cleanlab',
    version=__version__,
    license='GPLv3+',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    description = 'The standard package for machine learning with noisy labels and finding mislabeled data in Python.',
    url = 'https://github.com/cgnorthcutt/cleanlab',
    author = 'Curtis G. Northcutt',
    author_email = 'cgn@csail.mit.edu',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
      'Development Status :: 4 - Beta',

      'Intended Audience :: Developers',
      'Intended Audience :: Education',
      'Intended Audience :: Science/Research',
      'Intended Audience :: Information Technology',
      'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
      'Natural Language :: English',

      # We believe this package works will these versions, but we do not guarantee it!
      'Programming Language :: Python :: 2',
      'Programming Language :: Python :: 2.7',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.4',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
      'Programming Language :: Python :: 3.9',

      'Programming Language :: Python',
      'Topic :: Software Development',
      'Topic :: Scientific/Engineering',
      'Topic :: Scientific/Engineering',
      'Topic :: Scientific/Engineering :: Mathematics',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development',
      'Topic :: Software Development :: Libraries',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*',
    
    # What does your project relate to?
    keywords='machine_learning denoising classification weak_supervision learning_with_noisy_labels unsupervised_learning',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['img', 'examples']),
    
    # Include cleanlab license file.    
    include_package_data=True,
    package_data={
        "": ["LICENSE"],
    },
    license_files = ('LICENSE',),
    cmdclass = {'egg_info': egg_info_ex},

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy>=1.11.3', 'scikit-learn>=0.18', 'scipy>=1.1.0', 'tqdm>=4.53.0', ],
)
