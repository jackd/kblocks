import gin
import tensorflow_datasets as tfds

load = gin.external_configurable(tfds.load, module="tfds")
builder = gin.external_configurable(tfds.builder, module="tfds")
ReadConfig = gin.external_configurable(
    tfds.core.utils.read_config.ReadConfig, module="tfds"
)
DownloadConfig = gin.external_configurable(
    tfds.core.download.DownloadConfig, module="tfds"
)
