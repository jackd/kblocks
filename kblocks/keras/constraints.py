import tensorflow as tf

from kblocks.keras import wrap

loc = locals()
for k, v in wrap.wrapped_items(tf.keras.constraints, "tf.keras.constraints"):
    loc[k] = v

del loc, wrap
