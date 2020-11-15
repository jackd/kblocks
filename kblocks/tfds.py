import gin
import tensorflow as tf
import tensorflow_datasets as tfds

for func in (tfds.load, tfds.builder):
    gin.register(func, module="tfds")
    tf.keras.utils.register_keras_serializable("TFDS")(func)
