# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras Applications are canned architectures with pre-trained weights."""
import keras_applications
from tensorflow.python.keras import backend, engine, models, utils
from tensorflow.python.util import tf_inspect

from kblocks.keras import layers

# `get_submodules_from_kwargs` has been introduced in 1.0.5, but we would
# like to be able to handle prior versions. Note that prior to 1.0.5,
# `keras_applications` did not expose a `__version__` attribute.
if not hasattr(keras_applications, "get_submodules_from_kwargs"):
    # pylint: disable=no-member

    if (
        "engine"
        in tf_inspect.getfullargspec(keras_applications.set_keras_submodules)[0]
    ):
        keras_applications.set_keras_submodules(
            backend=backend, layers=layers, models=models, utils=utils, engine=engine
        )
    else:
        keras_applications.set_keras_submodules(
            backend=backend, layers=layers, models=models, utils=utils
        )
    # pylint: enable=no-member


def keras_modules_injection(base_fun):
    """Decorator injecting tf.keras replacements for Keras modules.

    Arguments:
        base_fun: Application function to decorate (e.g. `MobileNet`).

    Returns:
        Decorated function that injects keyword argument for the tf.keras
        modules required by the Applications.
    """

    def wrapper(*args, **kwargs):
        if hasattr(keras_applications, "get_submodules_from_kwargs"):
            kwargs["backend"] = backend
            if "layers" not in kwargs:
                kwargs["layers"] = layers
            kwargs["models"] = models
            kwargs["utils"] = utils
        return base_fun(*args, **kwargs)

    return wrapper


loc = locals()

for k, v in (
    ("DenseNet121", keras_applications.densenet.DenseNet121),
    ("DenseNet169", keras_applications.densenet.DenseNet169),
    ("DenseNet201", keras_applications.densenet.DenseNet201),
    ("InceptionResNetV2", keras_applications.inception_resnet_v2.InceptionResNetV2),
    ("InceptionV3", keras_applications.inception_v3.InceptionV3),
    ("MobileNet", keras_applications.mobilenet.MobileNet),
    ("MobileNetV2", keras_applications.mobilenet_v2.MobileNetV2),
    ("NASNetLarge", keras_applications.nasnet.NASNetLarge),
    ("NASNetMobile", keras_applications.nasnet.NASNetMobile),
    ("ResNet50", keras_applications.resnet50.ResNet50),
    ("VGG16", keras_applications.vgg16.VGG16),
    ("VGG19", keras_applications.vgg19.VGG19),
    ("Xception", keras_applications.xception.Xception),
):
    loc[k] = keras_modules_injection(v)

# make linters play nicely
DenseNet121 = loc["DenseNet121"]
DenseNet169 = loc["DenseNet169"]
DenseNet201 = loc["DenseNet201"]
InceptionResNetV2 = loc["InceptionResNetV2"]
InceptionV3 = loc["InceptionV3"]
MobileNet = loc["MobileNet"]
MobileNetV2 = loc["MobileNetV2"]
NASNetLarge = loc["NASNetLarge"]
NASNetMobile = loc["NASNetMobile"]
ResNet50 = loc["ResNet50"]
VGG16 = loc["VGG16"]
VGG19 = loc["VGG19"]
Xception = loc["Xception"]
del loc
