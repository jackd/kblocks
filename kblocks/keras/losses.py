from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from kblocks.keras import wrap

loc = locals()
for k, v in wrap.wrapped_items(tf.keras.losses, "tf.keras.losses"):
    loc[k] = v

del loc, wrap
