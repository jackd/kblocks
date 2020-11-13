import gin
import tensorflow as tf

CheckpointManager = gin.external_configurable(
    tf.train.CheckpointManager, module="tf.train"
)
Checkpoint = gin.external_configurable(tf.train.Checkpoint, module="tf.train")
