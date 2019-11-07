"""Supplies a default set of configurables from tensorflow.compat.v1."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sys import prefix

import tensorflow as tf
import gin
import imp
from types import ModuleType
from typing import Iterable, Any, Tuple

BLACKLIST = ('serialize', 'deserialize', 'get')


def renamed(child_mod: ModuleType, root_mod: ModuleType, new_root: str):
    root_name = root_mod.__name__
    child_name = child_mod.__name__
    if not child_name.startswith(root_name):
        raise ValueError(
            'child_name should start with root_name, but {} does not start '
            'with {}'.format(child_name, root_name))
    return '{}{}'.format(new_root, child_name[len(root_name):])


def wrapped_items(
        src_module: ModuleType,
        gin_module: str,
        blacklist=BLACKLIST,
) -> Iterable[Tuple[str, Any]]:
    for k in dir(src_module):
        v = getattr(src_module, k)
        if k not in blacklist and callable(v):
            yield (k, gin.external_configurable(v, name=k, module=gin_module))


def wrapped_module(dst_name,
                   src_module: ModuleType,
                   gin_module: str,
                   blacklist=BLACKLIST) -> ModuleType:
    mod = imp.new_module(dst_name)
    for k, v in wrapped_items(src_module, gin_module, blacklist):
        setattr(mod, k, v)
    return mod


loc = locals()
for _py_module, _suffix in (
    (tf.keras.callbacks, 'callbacks'),
    (tf.keras.constraints, 'constraints'),
    (tf.keras.layers, 'layers'),
    (tf.keras.losses, 'losses'),
    (tf.keras.metrics, 'metrics'),
    (tf.keras.optimizers, 'optimizers'),
    (tf.keras.regularizers, 'regularizers'),
):
    name = renamed(_py_module, tf.keras, __name__)
    loc[_suffix] = wrapped_module(name, _py_module,
                                  'tf.keras.{}'.format(_suffix))

loc['optimizers'].schedules = wrapped_module(
    renamed(tf.keras.optimizers.schedules, tf.keras, __name__),
    tf.keras.optimizers.schedules, 'tf.keras.optimizers.schedules')

# make linters play nicely
callbacks = loc['callbacks']
constraints = loc['constraints']
layers = loc['layers']
losses = loc['losses']
metrics = loc['metrics']
optimizers = loc['optimizers']
regularizers = loc['regularizers']

# clean up namespace
del loc
