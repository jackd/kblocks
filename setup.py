from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

setup(name='kblocks',
      version='0.0.1',
      description=(
          'gin-configured keras blocks for rapid prototyping and benchmarking'),
      url='https://github.com/jackd',
      author='Dominic Jack',
      author_email='thedomjack@gmail.com',
      license='MIT',
      packages=['kblocks'],
      install_requires=['gin-config'],
      zip_safe=True)
