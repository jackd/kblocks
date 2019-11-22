# [Injectable tf.keras Blocks](https://github.com/jackd/kblocks)

This package provides a rapid prototyping environment for running deep learning experiments. There are two main components:

- Injectable keras components - "blocks" - which are just regular `keras` classes wrapped in `gin.configurable`, including some custom ones I feel are missing from core `keras`; and
- a [framework](kblocks/framework/) submodule which provides various even-higher-than-keras-level interfaces for common tasks.

## Quick Start - without Injection

The following is an example of training a model without using dependency injection. It isn't the point of this package, but in order to appreciate the point you need to go through this process first.

### Define a [Problem](kblocks/framework/problems/core.py)

`Problem`s provide a `tf.data.Dataset`, information about its length, expected model outputs and losses and metrics used in `tf.keras.Model.compile`. [tensorflow_datasets](https://github.com/tensorflow/datasets) is a great source of datasets, and the [TfdsProblem](kblocks/framework/problems/tfds.py) is a convenient wrapper around this.

```python
import functools
import tensorflow as tf
import tensorflow_datasets as tfds
from kblocks.framework.problems import TfdsProblem

def cifar100_problem(data_dir='~/tensorflow_datasets'):
    def pre_batch_map(image: tf.Tensor, label: tf.Tensor, split=str):
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        if split == 'train':
            image = tf.image.random_flip_left_right(image)
        return image, label

    return  TfdsProblem(
        builder=tfds.builder('cifar100', data_dir=data_dir),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        split_map={'validation': 'test'},  # use test split for validation
        pre_batch_map={
            split: functools.partial(pre_batch_map, split=split)
            for split in ('train', 'test')
        },  # use different mapping functions before batching depending on split
        outputs_spec=tf.TensorSpec(shape=(None, 100), dtype=tf.float32),
    )
```

Note `outputs_spec` isn't necessarily the same as the labels spec. In this case, the dataset labels are class integers (`tf.TensorSpec(shape=(batch_size,), dtype=tf.int64)`) while the outputs of trained models should be logits (`tf.TensorSpec(shape=(batch_size, num_classes), dtype=tf.float32`)).

### Define a `model_fn`

We'll use a very basic CNN. Note you should not `compile` things here.

```python
def simple_cnn(image,
               outputs_spec,
               conv_filters=(16, 32),
               dense_units=(),
               activation='relu'):
    """
    Simple convolutional network architecture.

    Args:
        image: rank-4 float tensor
        outputs_spec: tf.TensorSpec corresponding to the desired output.
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

    num_classes = outputs_spec.shape[-1]
    logits = tf.keras.layers.Dense(num_classes)(x)

    return tf.keras.Model(inputs=image, outputs=logits)
```

`kblocks` currently uses a slight generalization of models called [Pipelines](kblocks/framework/pipelines/core.py), which combined a normal keras model with model-specific pre-batch and post-batch dataset mapping functions. In most cases this isn't necessary, in which case `ModelPipeline` works just fine

### Fit using a [Trainable](kblocks/framework/trainables)

```python
from kblocks.framework.trainables import Trainable
from kblocks.framework.pipelines import ModelPipeline


def simple_cnn_pipeline(features_spec, outputs_spec):
    return (ModelPipeline(features_spec, outputs_spec, model_fn=simple_cnn)


trainable = Trainable(
    problem=cifar100_problem(),
    pipeline_fn=simple_cnn_pipeline,
    optimizer_fn=tf.keras.optimizers.Adam,
    model_dir='/tmp/kblocks/cifar100/simple_cnn')

trainable.fit(batch_size, epochs=10)
```

### Explore results

Look around `/tmp/kblocks/cifar100/simple_cnn`. Some useful data is logged under `logs` (keras model summaries, training status etc), there are tensorboard summaries and saved models.

## Extending for the Command Line

The above is admittedly a fairly compliated way of doing something which is relatively simple in `keras` - so why `kblocks`? I'm glad you asked.

First, let's consider how we would make a command-line interface to specify the various parameters in `simple_cnn`.

```python
from absl import flags
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
def cifar100_problem(data_dir):
    ...


# blacklisting/whitelisting makes it clear what can and can't be configured
@gin.configurable(blacklist=['image', 'outputs_spec'])
def simple_cnn(image,
               outputs_spec,
               conv_filters=(16, 32),
               dense_units=(),
               activation='relu'):
    ...
```

We configure with a separate `.gin` file in the same directory. [gin-config](https://github.com/google/gin-config) is a powerful dependency injection

```gin
# in 'simple.gin'
import simple  # assumes simple.py contains above fns and is in same folder
import kblocks.keras_configurables  # exposes tf.keras.optimizers.Adam to gin
model_fn = @simple_cnn
problem = @cifar100_problem()
optimizer_fn = @tf.keras.optimizers.Adam

model_dir = '/tmp/kblocks/examples/cifar100/v0'

batch_size = 32
epochs = 10
```

This isn't a full configuration. On its own, lines like `batch_size=32` won't do anything - they bind a value to a `gin` macro, but there's nothing linking this bound value to the argument used in `fit`. The idea is to use it in conjunction with various base configuration files in [\$KB_CONFIG](kblocks/configs) (`kblocks` defines this environment variable locally so you can use it from the command-line). To train, we just call `kblocks.__main__` with any base configuration and customized ones.

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
@gin.configurable(blacklist=['image', 'outputs_spec'])
def simple_cnn(image,
               outputs_spec,
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

    num_classes = outputs_spec.shape[-1]
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

@gin.configurable(blacklist=['image', 'outputs_spec'])
def simple_cnn(image,
               outputs_spec,
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

    num_classes = outputs_spec.shape[-1]
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
