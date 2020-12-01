import os

import gin
import tensorflow as tf

from kblocks.gin_utils.config import try_register_config_dir

gin.constant("AUTOTUNE", tf.data.experimental.AUTOTUNE)

KB_CONFIG_DIR = os.path.realpath(os.path.dirname(__file__))
try_register_config_dir("KB_CONFIG", KB_CONFIG_DIR)
