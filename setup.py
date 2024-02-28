from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info

# To use a consistent encoding
from codecs import open
from os import path


class egg_info_ex(egg_info):
    """Includes license file into `.egg-info` folder."""

    def run(self):
        # don't duplicate license into `.egg-info` when building a distribution
        if not self.distribution.have_run.get("install", True):
            # `install` command is in progress, copy license
            self.mkpath(self.egg_info)
            self.copy_file("LICENSE", self.egg_info)

        egg_info.run(self)


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get version number and store it in __version__
exec(open("cleanlab/version.py").read())

DATALAB_REQUIRE = [
    # Mainly for Datalab's data storage class.
    # Still some type hints that require datasets
    "datasets>=2.7.0",
]

IMAGE_REQUIRE = DATALAB_REQUIRE + ["cleanvision>=0.3.6"]

EXTRAS_REQUIRE = {
    "datalab": DATALAB_REQUIRE,
    "image": IMAGE_REQUIRE,
    "all": ["matplotlib>=3.5.1"],
}
EXTRAS_REQUIRE["all"] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

setup(
    name="cleanlab",
    version=__version__,
    license="AGPLv3+",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="The standard package for data-centric AI, machine learning with label errors, "
    "and automatically finding and fixing dataset issues in Python.",
    url="https://cleanlab.ai",
    project_urls={
        "Documentation": "https://docs.cleanlab.ai",
        "Bug Tracker": "https://github.com/cleanlab/cleanlab/issues",
        "Source Code": "https://github.com/cleanlab/cleanlab",
    },
    author="Cleanlab Inc.",
    author_email="team@cleanlab.ai",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Natural Language :: English",
        # We believe this package works will these versions, but we do not guarantee it!
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    # What does your project relate to?
    keywords="machine_learning data_cleaning confident_learning classification weak_supervision "
    "learning_with_noisy_labels unsupervised_learning datacentric_ai, datacentric",
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=[]),
    # Include cleanlab license file.
    include_package_data=True,
    package_data={
        "": ["LICENSE"],
    },
    license_files=("LICENSE",),
    cmdclass={"egg_info": egg_info_ex},
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/
    install_requires=[
        "numpy>=1.22.0",
        "scikit-learn>=1.1",
        "tqdm>=4.53.0",
        "pandas>=1.4.0",
        "termcolor>=2.4.0",
    ],
    extras_require=EXTRAS_REQUIRE,
)
