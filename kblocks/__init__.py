import sys
import tensorflow as tf
from kblocks.version import __version__

if (sys.version_info < (3, 6)):
    raise NotImplementedError('Only python 3.6 onwards supported')

if not tf.version.VERSION.startswith('2'):
    raise NotImplementedError('Only tensorflow 2 supported')

__all__ = [
    '__version__',
]
