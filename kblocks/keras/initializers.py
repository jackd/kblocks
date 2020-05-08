from __future__ import absolute_import, division, print_function

import tensorflow as tf

from kblocks.keras import wrap

loc = locals()
for k, v in wrap.wrapped_items(tf.keras.initializers, "tf.keras.initializers"):
    loc[k] = v

del loc, wrap
