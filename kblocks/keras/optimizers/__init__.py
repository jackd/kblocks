import tensorflow as tf

from kblocks.keras import wrap
from kblocks.keras.optimizers import schedules

loc = locals()
for k, v in wrap.wrapped_items(tf.keras.optimizers, "tf.keras.optimizers"):
    loc[k] = v

del loc, wrap
