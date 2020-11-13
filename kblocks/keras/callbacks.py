import tensorflow as tf

from kblocks.keras import wrap

loc = locals()
for k, v in wrap.wrapped_items(tf.keras.callbacks, "tf.keras.callbacks"):
    loc[k] = v

# make linter shut up
TensorBoard = loc["TensorBoard"]

del loc, wrap
