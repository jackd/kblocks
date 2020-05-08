from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup
import os
import sys

version_path = os.path.join(os.path.dirname(__file__), "kblocks")
sys.path.append(version_path)
from version import __version__  # pylint: disable=no-name-in-module

with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as fp:
    install_requires = fp.read().split("\n")

setup(
    name="kblocks",
    description=("gin-configured keras blocks for rapid prototyping and benchmarking"),
    url="https://github.com/jackd",
    author="Dominic Jack",
    author_email="thedomjack@gmail.com",
    license="MIT",
    packages=["kblocks"],
    install_requires=install_requires,
    zip_safe=True,
    version=__version__,
)
