import tensorflow as tf

from kblocks.keras import wrap

loc = locals()
for k, v in wrap.wrapped_items(
    tf.keras.optimizers.schedules, "tf.keras.optimizers.schedules"
):
    loc[k] = v

del loc, wrap
