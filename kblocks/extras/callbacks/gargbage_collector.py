import gc

import tensorflow as tf


class GarbageCollector(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch: int, logs=None):
        gc.collect()
