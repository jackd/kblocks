import sys
import tensorflow as tf

if (sys.version_info < (3, 6)):
    raise NotImplementedError('Only python 3.6 onwards supported')

if not tf.version.VERSION.startswith('2'):
    raise NotImplementedError('Only tensorflow 2 supported')

from kblocks import metrics
from kblocks import layers
from kblocks import ops
from kblocks import callbacks
from kblocks import schedules
from kblocks import constraints

__all__ = ['metrics', 'layers', 'ops', 'callbacks', 'schedules', 'constraints']
