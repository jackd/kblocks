from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional
import tensorflow as tf
from kblocks.extras.layers import BiasAdd
from kblocks.extras.layers import Scale
import gin


@gin.configurable(module='kb.extras.layers')
def Denormalization(bias: Optional[BiasAdd] = None,
                    scale: Optional[Scale] = None):
    if bias is None:
        bias = BiasAdd()
    if scale is None:
        scale = Scale()
    return tf.keras.models.Sequential([scale, bias])
