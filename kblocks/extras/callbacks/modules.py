from typing import Optional

import gin
import tensorflow as tf
from absl import logging

from kblocks.keras import callbacks as base
from kblocks.serialize import register_serializable
from kblocks.utils import super_signature


def variable_property(name: str, dtype: tf.DType, doc: Optional[str] = None, **kwargs):
    """
    Get a property that wraps `tf.Variable` assignment.

    Useful for augmenting a base class to save values in `tf.Variable`s rather than
    as attributes.

    Example usage:
    ```python
    class Foo:
        def __init__(self, a, b):
            self.a = a
            self.b = b


    class FooModule(Foo, tf.Module):
        def __init__(self, a, b, name=None):
            tf.Module.__init__(self, name=name)
            Foo.__init__(self, a, b)

        a = variable_property("a", tf.int64)
        b = variable_property("b", tf.int64)

    foo = FooModule(2, 3)
    assert foo.a == 2
    print(list(foo._flatten()))

    chkpt = tf.train.Checkpoint(foo=foo)
    path = chkpt.save("/tmp/foo")
    foo.a = 10
    chkpt.restore(path)
    assert foo.a == 2
    ```
    """
    attr_name = f"_variable_{name}"

    def getx(self):
        return tf.keras.backend.get_value(getattr(self, attr_name))

    @tf.Module.with_name_scope
    def setx(self, value):
        variable = getattr(self, attr_name, None)
        if variable is None:
            variable = tf.Variable(value, dtype=dtype, name=name, **kwargs)
            setattr(self, attr_name, variable)
        else:
            variable.assign(value)

    def delx(self):
        delattr(self, attr_name)

    return property(getx, setx, delx, doc)


class CallbackModule(tf.Module):
    def set_model(self, model: tf.keras.Model):
        # pylint: disable=protected-access
        old_model = getattr(self, "model", None)
        if old_model is not None:
            del old_model._callbacks[self.name]
        if not hasattr(model, "_callbacks"):
            model._callbacks = {}

        callbacks = model._callbacks
        # pylint: enable=protected-access
        assert self.name not in callbacks
        callbacks[self.name] = self
        self.model = model  # pylint: disable=attribute-defined-outside-init


@gin.configurable(module="kb.callbacks")
@register_serializable
@super_signature
class ReduceLROnPlateauModule(base.ReduceLROnPlateau, CallbackModule):
    def __init__(self, **kwargs):
        CallbackModule.__init__(self, name=None)
        self._supports_tf_logs = True
        self._config = dict(kwargs)
        base.ReduceLROnPlateau.__init__(self, **self._config)

    def get_config(self):
        return dict(self._config)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    best = variable_property("best", tf.float32, trainable=False)
    wait = variable_property("wait", tf.int64, trainable=False)
    cooldown_counter = variable_property("cooldown_counter", tf.int64, trainable=False)


@gin.configurable(module="kb.callbacks")
@register_serializable
@super_signature
class EarlyStoppingModule(base.EarlyStopping, CallbackModule):
    def __init__(self, **kwargs):
        CallbackModule.__init__(self)
        self._config = dict(kwargs)
        self._best_weights = None
        base.EarlyStopping.__init__(self, **self._config)

    best = variable_property("best", tf.float32, trainable=False)
    wait = variable_property("wait", tf.int64, trainable=False)
    stopped_epoch = variable_property("stopped_epoch", tf.int64, trainable=False)

    def get_config(self):
        return dict(self._config)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def best_weights(self):
        weights = self._best_weights
        if weights is None:
            return weights
        return [tf.keras.backend.get_value(w) for w in weights]

    @best_weights.setter
    @tf.Module.with_name_scope
    def best_weights(self, value):
        if value is None:
            self._best_weights = None
            return
        if self._best_weights is None:
            self._best_weights = [
                tf.Variable(w, name=f"best_weights-{i}", trainable=False)
                for i, w in enumerate(value)
            ]
        else:
            assert len(value) == len(self._best_weights)
            for src, dst in zip(value, self._best_weights):
                dst.assign(src)

    def on_epoch_begin(self, epoch, logs=None):
        del logs
        if epoch == self.stopped_epoch and self.stopped_epoch > 0:
            logging.info("EarlyStoppingModule has already stopped training.")
            self.model.stop_training = True


def get(identifier) -> tf.keras.callbacks.Callback:
    if isinstance(identifier, tf.keras.callbacks.Callback):
        return identifier

    callback = tf.keras.utils.deserialize_keras_object(identifier)
    if not isinstance(callback, tf.keras.callbacks.Callback):
        raise ValueError(f"Invalid callback: {callback}")
    return callback
