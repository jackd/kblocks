import sys
import tensorflow as tf

if (sys.version_info < (3, 6)):
    raise NotImplementedError('Only python 3.6 onwards supported')

if not tf.version.VERSION.startswith('2'):
    raise NotImplementedError('Only tensorflow 2 supported')
