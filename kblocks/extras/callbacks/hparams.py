from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gin
from tensorboard.plugins.hparams import api as hp
from kblocks.scope import Scope


class HParamsBuilder(object):
    def __init__(self):
        self._hparams = {}

    def register(self, key: str, domain: hp.Domain, value):
        self._hparams[hp.HParam(key, domain)] = value

    @property
    def hparams(self):
        return self._hparams

    def keras_callback(self, log_dir):
        return hp.KerasCallback(log_dir, self.hparams)


scope = Scope[HParamsBuilder](HParamsBuilder(), name="hparams")
get_default = scope.get_default


def register(key, domain, value):
    return get_default().register(key, domain, value)


def get_hparams():
    return get_default().hparams


@gin.configurable(module="kb.callbacks")
def hp_callback(log_dir, hparams=None):
    return get_default().keras_callback(log_dir)


HPCallback = hp_callback
