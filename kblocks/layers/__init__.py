from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin
from kblocks.layers.zero_init import ZeroInit

# Configurable versions of base layers to be used from code.
BatchNormalization = gin.external_configurable(
    tf.keras.layers.BatchNormalization, module='kb.layers')
Convolution1D = gin.external_configurable(tf.keras.layers.Convolution1D,
                                          module='kb.layers')
Convolution2D = gin.external_configurable(tf.keras.layers.Convolution2D,
                                          module='kb.layers')
Convolution3D = gin.external_configurable(tf.keras.layers.Convolution3D,
                                          module='kb.layers')
Dense = gin.external_configurable(tf.keras.layers.Dense, module='kb.layers')
Dropout = gin.external_configurable(tf.keras.layers.Dropout, module='kb.layers')

Lambda = tf.keras.layers.Lambda
Input = tf.keras.layers.Input

# class Lambda(tf.keras.layers.Lambda):

#     def __init__(self, *args, **kwargs):
#         super(Lambda, self).__init__(*args, **kwargs)
#         print(self.name)
#         # if self.name == 'lambda_64':
#         #     raise Exception()

# def Input(*args, **kwargs):
#     out = tf.keras.layers.Input(*args, **kwargs)
#     print(out._keras_history.layer.name, type(out))
#     # if out._keras_history.layer.name == 'input_51':
#     #     raise Exception()
#     return out

__all__ = [
    'BatchNormalization',
    'Convolution1D',
    'Convolution2D',
    'Convolution3D',
    'Dense',
    'Dropout',
    'Lambda',
    'Input',
    'ZeroInit',
]
