from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf
import gin
from typing import Sequence, Optional


@gin.configurable(module='kb')
def get_optimizer_options(layout_optimizer=True,
                          constant_folder=True,
                          shape_optimization=True,
                          remapping=True,
                          arithmetic_optimization=True,
                          dependency_optimization=True,
                          loop_optimization=True,
                          function_optimization=True,
                          debug_stripper=True,
                          disable_model_pruning=False,
                          scoped_allocator_optimization=True,
                          pin_to_host_optimization=True,
                          implementaion_selector=True,
                          auto_mixed_precision=False,
                          disable_meta_optimizer=False,
                          min_graph_nodes=0):
    return dict(
        layout_optimizer=layout_optimizer,
        constant_folder=constant_folder,
        shape_optimization=shape_optimization,
        remapping=remapping,
        arithmetic_optimization=arithmetic_optimization,
        dependency_optimization=dependency_optimization,
        loop_optimization=loop_optimization,
        function_optimization=function_optimization,
        debug_stripper=debug_stripper,
        disable_model_pruning=disable_model_pruning,
        scoped_allocator_optimization=scoped_allocator_optimization,
        pin_to_host_optimization=pin_to_host_optimization,
        implementaion_selector=implementaion_selector,
        auto_mixed_precision=auto_mixed_precision,
        disable_meta_optimizer=disable_meta_optimizer,
        min_graph_nodes=min_graph_nodes,
    )


@gin.configurable(module='kb')
class TfConfig(object):

    def __init__(self,
                 allow_growth: bool = True,
                 visible_devices: Optional[Sequence[int]] = None,
                 jit: Optional[bool] = None,
                 optimizer_options: Optional[dict] = None):
        self.optimizer_options = optimizer_options
        self.allow_growth = allow_growth
        self.visible_devices = visible_devices
        self.jit = jit

    def configure(self):
        if self.visible_devices is not None:
            devices = tf.config.experimental.get_visible_devices('GPU')
            devices = {d.name.split(':')[-1]: d for d in devices}
            devices = [devices[str(d)] for d in self.visible_devices]
            tf.config.experimental.set_visible_devices(devices,
                                                       device_type='GPU')
        try:
            for device in tf.config.experimental.get_visible_devices('GPU'):
                tf.config.experimental.set_memory_growth(
                    device, self.allow_growth)
        except Exception:
            logging.info('Failed to set memory growth to {}'.format(
                self.allow_growth))
        if self.optimizer_options is not None:
            tf.config.optimizer.set_experimental_options(self.optimizer_options)
        if self.jit is not None:
            tf.config.optimizer.set_jit(self.jit)

    def get_config(self):
        return dict(allow_growth=self.allow_growth,
                    visible_devices=self.visible_devices,
                    jit=self.jit,
                    optimizer_options=self.optimizer_options)

    @classmethod
    def from_config(cls, config):
        return TfConfig(**config)
