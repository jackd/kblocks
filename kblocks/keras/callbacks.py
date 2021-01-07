import gin
import tensorflow as tf

from kblocks.keras import wrap
from kblocks.utils import super_signature

loc = locals()
for k, v in wrap.wrapped_items(
    tf.keras.callbacks,
    "tf.keras.callbacks",
    blacklist=wrap.BLACKLIST + ("LearningRateScheduler", "ReduceLROnPlateau"),
):
    loc[k] = v

# make linter shut up
TensorBoard = loc["TensorBoard"]
CSVLogger = loc["CSVLogger"]
EarlyStopping = loc["EarlyStopping"]
BackupAndRestore = gin.external_configurable(
    tf.keras.callbacks.experimental.BackupAndRestore, module="tf.keras.callbacks"
)

del loc, wrap

# add _supports_tf_logs = True
# Github issue: https://github.com/tensorflow/tensorflow/issues/45895


@gin.configurable(module="tf.keras.callbacks")
@super_signature
class ReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._supports_tf_logs = True


@gin.configurable(module="tf.keras.callbacks")
@super_signature
class LearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._supports_tf_logs = True
