import os
from pathlib import Path

from setuptools import find_packages, setup

# We follow Semantic Versioning (https://semver.org/)
_MAJOR_VERSION = "0"
_MINOR_VERSION = "3"
_PATCH_VERSION = "0"


def glob_fix(package_name, glob):
    package_path = Path(os.path.join(os.path.dirname(__file__), package_name)).resolve()
    return [str(path.relative_to(package_path)) for path in package_path.glob(glob)]


with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as fp:
    install_requires = fp.read().split("\n")

setup(
    name="kblocks",
    description="gin-configured keras blocks for rapid prototyping and benchmarking",
    url="https://github.com/jackd/kblocks",
    author="Dominic Jack",
    author_email="thedomjack@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=install_requires,
    package_data={"kblocks": glob_fix("kblocks", "configs/**/*.gin")},
    zip_safe=True,
    python_requires=">=3.6",
    version=".".join(
        [
            _MAJOR_VERSION,
            _MINOR_VERSION,
            _PATCH_VERSION,
        ]
    ),
)
