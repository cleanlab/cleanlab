from setuptools import setup
from setuptools.command.egg_info import egg_info

# To use a consistent encoding
from codecs import open


class egg_info_ex(egg_info):
    """Includes license file into `.egg-info` folder."""

    def run(self):
        # don't duplicate license into `.egg-info` when building a distribution
        if not self.distribution.have_run.get("install", True):
            # `install` command is in progress, copy license
            self.mkpath(self.egg_info)
            self.copy_file("LICENSE", self.egg_info)

        egg_info.run(self)


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
    version=__version__,
    cmdclass={"egg_info": egg_info_ex},
    extras_require=EXTRAS_REQUIRE,
)
