from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from kblocks.keras import wrap

loc = locals()
for k, v in wrap.wrapped_items(tf.keras.layers, 'tf.keras.layers'):
    loc[k] = v

# make linters shut up
BatchNormalization = loc['BatchNormalization']
Dense = loc['Dense']
Convolution1D = loc['Convolution1D']
Conv1D = loc['Conv1D']
Convolution2D = loc['Convolution2D']
Conv2D = loc['Conv2D']
Convolution3D = loc['Convolution3D']
Conv3D = loc['Conv3D']
Dropout = loc['Dropout']

del loc, wrap
