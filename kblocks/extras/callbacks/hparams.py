from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gin
from tensorboard.plugins.hparams import api as hp


class HParamsScope(object):
    _stack = []

    def __enter__(self):
        HParamsScope._stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        x = HParamsScope._stack.pop()
        assert (x is self)

    def __init__(self):
        self._hparams = {}

    def register(self, key: str, domain: hp.Domain, value):
        self._hparams[hp.HParam(key, domain)] = value

    @property
    def hparams(self):
        return self._hparams

    def keras_callback(self, log_dir):
        return hp.KerasCallback(log_dir, self.hparams)


def get_default_scope():
    return HParamsScope._stack[-1]


# add default scope
HParamsScope._stack.append(HParamsScope())


def register(key, domain, value):
    return get_default_scope().register(key, domain, value)


def get_hparams():
    return get_default_scope().hparams


@gin.configurable(module='kb.callbacks')
def hp_callback(log_dir, hparams=None):
    return get_default_scope().keras_callback(log_dir)


HPCallback = hp_callback
