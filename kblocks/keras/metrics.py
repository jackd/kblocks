import tensorflow as tf

from kblocks.keras import wrap

loc = locals()
for k, v in wrap.wrapped_items(tf.keras.metrics, "tf.keras.metrics"):
    loc[k] = v

del loc, wrap
