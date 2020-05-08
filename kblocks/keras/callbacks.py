from __future__ import absolute_import, division, print_function

import tensorflow as tf

from kblocks.keras import wrap

loc = locals()
for k, v in wrap.wrapped_items(tf.keras.callbacks, "tf.keras.callbacks"):
    loc[k] = v

del loc, wrap
