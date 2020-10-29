# [Injectable tf.keras Blocks](https://github.com/jackd/kblocks)

This package provides a rapid prototyping environment for running deep learning experiments. There are two main components:

- Injectable keras components - "blocks" - which are just regular `keras` classes wrapped in `gin.configurable`, including some custom ones I feel are missing from core `keras`; and
- a [framework](kblocks/framework/) submodule which provides various even-higher-than-keras-level interfaces for common tasks.

## Installation

```bash
pip install tensorflow>=2.3  # not included in requirements.txt - could be tf-nightly
git clone https://github.com/jackd/kblocks.git
pip install -e kblocks
```

## Quick Start - without Injection

The following is an example of training a model without using dependency injection. It isn't the point of this package, but in order to appreciate the point you need to go through this process first.

### Define a [Source](kblocks/framework/sources/core.py)

`Source`s provide `tf.data.Dataset`s and meta-data. [tensorflow_datasets](https://github.com/tensorflow/datasets) is a great source of datasets, and the `TfdsSource` is a convenient wrapper around `tfds.DatasetBuilder`s. Most data pipelines involve some degree of mapping, batching and shuffling, and [PipelinedSource](kblocks/framework/sources/pipelined.py) provide these.

```python
import functools
import tensorflow as tf
import tensorflow_datasets as tfds
from kblocks.framework.sources import TfdsSource
from kblocks.framework.batchers import RectBatcher

def cifar100_source(data_dir='~/tensorflow_datasets'):
    def pre_batch_map(image: tf.Tensor, label: tf.Tensor, split=str):
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        if split == 'train':
            image = tf.image.random_flip_left_right(image)
        return image, label

     return PipelinedSource(
        source=TfdsSource(
            "cifar100",
            # use the last 10% of the official training set for validation
            split_map={"train": "train[:90]", "validation": "train[90:]"},
            meta=dict(num_classes=100),  # passed as **kwargs to `simple_cnn` below
        ),
        batcher=RectBatcher(batch_size),
        pre_batch_map=pre_batch_map,
        shuffle_buffer=shuffle_buffer,
    )
```

### Define a `model_fn`

We'll use a very basic CNN. Note you should not `compile` things here.

```python
def simple_cnn(image,
               num_classes: int,
               conv_filters=(16, 32),
               dense_units=(256,),
               activation="relu"):
    """
    Simple convolutional network architecture.

    Args:
        image: rank-4 float tensor
        num_classes: number of classes in the output.
        conv_filters: sequence of ints describing number of filters in each
            convolutional layer
        dense_units: sequence of ints describing number of units in each
            dense layer after the final flatten.
        activation: string or tf function used between layers.

    Returns:
        Uncompiled `tf.keras.Model`.
    """
    x = image
    for f in conv_filters:
        x = tf.keras.layers.Convolution2D(f, 3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Flatten()(x)
    for u in dense_units:
        x = tf.keras.layers.Dense(u)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)

    logits = tf.keras.layers.Dense(num_classes)(x)

    return tf.keras.Model(inputs=image, outputs=logits)
```

### Fit using a [Trainable](kblocks/framework/trainables)

```python
import functools

from absl import flags

import kblocks.framework.compilers
import simple  # assumes simple.py contains above fns and is in same folder
from kblocks.framework.compilers import compile_classification_model
from kblocks.framework.trainable import base_trainable, fit

source = cifar100_source()
optimizer = tf.keras.optimizers.Adam()
trainable = base_trainable(
    source=source,
    model_fn=simple_cnn,
    compiler=functools.partial(compile_classification_model, optimizer=optimizer),
    model_dir="/tmp/kblocks/examples/cifar100"
)

fit(trainable, epochs=30)
```

### Explore results

Look around `/tmp/kblocks/cifar100/simple_cnn`. Some useful data is logged under `logs` (keras model summaries, training status etc), there are tensorboard summaries and saved models.

## Extending for the Command Line

The above is admittedly a fairly compliated way of doing something which is relatively simple in `keras` - so why `kblocks`? I'm glad you asked.

First, let's consider how we would make a command-line interface to specify the various parameters in `simple_cnn`.

```python
flags.DEFINE_list(
    'conv_filters', default=[16, 32],
    help='sequence of ints describing number of filters in each '
         'convolutional layer')
flags.DEFINE_string('activation', default='relu', help='network activations')
...
```

There's a lot of redundancy here in terms of documentation and default values. If we wanted to give the user a different `model_fn`, we'd need to include separate command line args for each, even though only one set would ever be used for a given run. We'd also have to write the code that links these command line args to the function args, and that's error prone.

## Using Dependency Injection

Making the above code injectable is as simple as adding a `gin.configurable` wrapper around our methods.

```python
@gin.configurable()
def cifar100_source(data_dir):
    ...


# blacklisting/whitelisting makes it clear what can and can't be configured
@gin.configurable(blacklist=['image', 'num_classes'])
def simple_cnn(image,
               num_classes: int,
               conv_filters=(16, 32),
               dense_units=(),
               activation='relu'):
    ...
```

We configure with a separate `.gin` file in the same directory. [gin-config](https://github.com/google/gin-config) is a powerful dependency injection

```gin

model_fn = @simple_cnn
source = @cifar100_source()
compiler = @compile_classification_model

compile_classification_model.optimizer = %optimizer
optimizer = @tf.keras.optimizers.Adam()

cifar100_source.batch_size = %batch_size

model_dir = '/tmp/kblocks/examples/cifar100/v0'

batch_size = 32
epochs = 10
```

This isn't a full configuration. On its own, lines like `epochs=10` won't do anything - they bind a value to a `gin` macro, but there's nothing linking this bound value to the argument used in `fit`. The idea is to use it in conjunction with various base configuration files in [\$KB_CONFIG](kblocks/configs) (`kblocks` defines this environment variable locally so you can use it from the command-line as a string, e.g. `'$KB_CONFIG/fit'`). To train, we just call `kblocks.__main__` with [$KB_CONFIG/fit.gin](kblocks/configs/fit.gin) and customized ones.

```bash
python -m kblocks '$KB_CONFIG/fit.gin' simple.gin
```

To customize parameters, we can just use new-line separated `gin` bindings

```bash
python -m kblocks '$KB_CONFIG/fit.gin' simple.gin --bindings='
simple_cnn.dense_units = (100,)
simple_cnn.activation = "sigmoid"
model_dir = "/tmp/kblocks/examples/cifar100/v1"
'
```

You can use any number of gin files. For example, the above could be refactored by creating a separate gin file in the same directory.

```gin
# in alt_arch.gin
simple_cnn.dense_units = (100,)
simple_cnn.activation = "sigmoid"
```

```gin
# in decayed_adam.gin
import kblocks.keras_configurables
optimizer_fn = @tf.keras.optimizers.Adam
Adam.learning_rate = @tf.keras.optimizers.schedules.ExponentialDecay()
ExponentialDecay.initial_learning_rate = 1e-3
ExponentialDecay.decay_steps = 1000
```

```bash
# trailing .gin extensions to positional CL args are optional
python -m kblocks '$KB_CONFIG/fit' simple alt_arch decayed_adam --bindings='model_dir="/tmp/kblocks/examples/cifar100/v3'
```

Thanks to `gin` and `kblocks.keras_configurables` we get a command line interface to all of `keras` for free!

### Configurables within Code

Let's say we wanted to add regularization. We might be tempted to try something like

```gin
tf.keras.layers.Convolution2D.kernel_regularizer = %regularizer
tf.keras.layers.Dense.kernel_regularizer = %regularizer
regularizer = @tf.keras.regularizers.l2()
tf.keras.regularizers.l2.l = 5e-4
```

Unfortunately, this won't work as we're expecting. This is because the places we are calling `Convolution2D` and `Dense` are using native `keras` methods, not `gin` wrapped versions that would come about. We can get around this in two ways by modifying the code.

#### Option 1: Accept Custom Convolution2D/Dense Implementations

```python
@gin.configurable(blacklist=['image', 'num_classes'])
def simple_cnn(image,
               num_classes: int,
               conv_filters=(16, 32),
               dense_units=(),
               activation='relu',
               Convolution2D=tf.keras.layers.Convolution2D,
               Dense=tf.keras.layers.Dense):
    """Simple convolutional network architecture."""
    x = image
    for f in conv_filters:
        x = Convolution2D(f, 3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Flatten()(x)
    for u in dense_units:
        x = Dense(u)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)

    logits = Dense(num_classes)(x)

    return tf.keras.Model(inputs=image, outputs=logits)
```

```gin
import kblocks.keras_configurables
simple_cnn.Convolution2D = @tf.keras.layers.Convolution2D
simple_cnn.Dense = @tf.keras.layers.Dense
```

#### Option 2: Use Configurable Versions in the Function

```python
Convolution2D = gin.external_configurable(tf.keras.layers.Convolution2D)
Dense = gin.external_configurable(tf.keras.layers.Dense)
BatchNormalization = gin.external_configurable(tf.keras.layers.BatchNormalization)

@gin.configurable(blacklist=['image', 'num_classes'])
def simple_cnn(image,
               num_classes,
               conv_filters=(16, 32),
               dense_units=(),
               activation='relu'):
    """Simple convolutional network architecture."""
    x = image
    for f in conv_filters:
        x = Convolution2D(f, 3)(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Flatten()(x)
    for u in dense_units:
        x = Dense(u)(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)

    logits = Dense(num_classes)(x)

    return tf.keras.Model(inputs=image, outputs=logits)
```

Now we can also do things like modify the batch normalization momentum.

```gin
BatchNormalization.momentum = 0.9
```

The above is so common most of these `external_configurables` are available in [kblocks.keras.layers](kblocks/layers/__init__.py).

```python
from kblocks.keras.layers import Dense, Convolution2D, BatchNormalization
```

Note that using the objects output by `external_cofigurable` from code is different to using the argument passed in. Base implementations called from code (e.g. `tf.keras.layers.Dense`) will not have their default parameters changed by `gin` bindings (though `kblocks.keras.layers.Dense` will).

## Further Reading

Go check out the [gin user guide](https://github.com/google/gin-config/blob/master/docs/index.md) for more examples of how best to use this powerful framework. Happy configuring!

## Reproducibility

The aim is for training with [Trainable.fit](kblocks/framework/trainable.py) to lead to reproducible results when. This is currently possible if training is performed in one single step, though will fail if run over multiple calls (with checkpoints saved). In order to achieve this:

1. training must be performed in a single step (i.e. no restarting from earlier `fit`s see below);
2. `TfConfig.seed` must be configured;
3. data augmentation functions must use `tf.random.Generator` ops, rather than `tf.random` ops, and the `tf.random.Generator`(s) used should be made available in `DataSource.modules` to ensure their state is saved during training. Additionally, `tf.data.Dataset.map` calls with random ops must use `num_parallel_calls=1`; and
4. if using `kblocks.extras.cache.SnapshotManager`, `preprocess` must be `False`.

Current limitations to the (1) are:

- how do we save `tf.data.Dataset.shuffle` state?
- how do we make `tf.keras.layers.Dropout` reproducible over split runs? Custom implementation using `tf.random.Generator`?

## Projects using `kblocks`

- Implementations from [Sparse Convolutions on Continuous Domains](https://github.com/jackd/sccd.git):
  - [Point Cloud Network](https://github.com/jackd/pcn.git)
  - [Event Convolution Network](https://github.com/jackd/ecn.git)

## TODO

- seeding / reproducible results: with dataset shuffling / dropout
  - shuffle state?
  - custom dropout layer?
- refactor [polynomials](kblocks/ops/polynomials) into separate repo?
