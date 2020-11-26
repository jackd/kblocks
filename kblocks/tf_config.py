import os
from typing import Optional, Sequence

import gin
import tensorflow as tf
from absl import logging


@gin.configurable(module="kb")
def get_optimizer_options(
    layout_optimizer=True,
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
    min_graph_nodes=0,
):
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


@gin.configurable(module="kb")
class TfConfig:
    """GPU and JIT configurations for tensoflow."""

    def __init__(
        self,
        allow_growth: bool = True,
        visible_devices: Optional[Sequence[int]] = None,
        jit: Optional[bool] = None,
        optimizer_options: Optional[dict] = None,
        log_device_placement: Optional[bool] = None,
        seed: Optional[int] = None,
        global_rng_seed: Optional[int] = None,
        inter_op_parallelism_threads: int = 0,
        intra_op_parallelism_threads: int = 0,
        deterministic_ops: bool = False,
    ):
        self.optimizer_options = optimizer_options
        self.allow_growth = allow_growth
        self.visible_devices = visible_devices
        self.jit = jit
        self.log_device_placement = log_device_placement
        self.seed = seed
        self.global_rng_seed = global_rng_seed
        self.inter_op_parallelism_threads = inter_op_parallelism_threads
        self.intra_op_parallelism_threads = intra_op_parallelism_threads
        self.deterministic_ops = deterministic_ops

    def configure(self):
        if self.visible_devices is not None:
            devices = tf.config.experimental.get_visible_devices("GPU")
            devices = {d.name.split(":")[-1]: d for d in devices}
            devices = [devices[str(d)] for d in self.visible_devices]
            tf.config.experimental.set_visible_devices(devices, device_type="GPU")
        try:
            for device in tf.config.experimental.get_visible_devices("GPU"):
                tf.config.experimental.set_memory_growth(device, self.allow_growth)
        except (ValueError, RuntimeError):
            logging.info("Failed to set memory growth to {}".format(self.allow_growth))
        if self.optimizer_options is not None:
            tf.config.optimizer.set_experimental_options(self.optimizer_options)
        if self.jit is not None:
            tf.config.optimizer.set_jit(self.jit)

        dp = self.log_device_placement
        if dp is not None:
            tf.debugging.set_log_device_placement(dp)
        tf.config.threading.set_inter_op_parallelism_threads(
            self.inter_op_parallelism_threads
        )
        tf.config.threading.set_intra_op_parallelism_threads(
            self.intra_op_parallelism_threads
        )
        # tf.keras.backend.clear_session()
        if self.seed is not None:
            tf.random.set_seed(self.seed)
        if self.global_rng_seed is not None:
            tf.random.get_global_generator().reset_from_seed(self.global_rng_seed)
            # worse results when we create a new rng as below - no idea why ??
            # tf.random.set_global_generator(
            #     tf.random.Generator.from_seed(self.global_rng_seed)
            # )
        os.environ["TF_DETERMINISTIC_OPS"] = "1" if self.deterministic_ops else "0"

    def get_config(self):
        return dict(
            allow_growth=self.allow_growth,
            visible_devices=self.visible_devices,
            jit=self.jit,
            optimizer_options=self.optimizer_options,
            log_device_placement=self.log_device_placement,
            seed=self.seed,
            global_rng_seed=self.global_rng_seed,
            inter_op_parallelism_threads=self.inter_op_parallelism_threads,
            intra_op_parallelism_threads=self.intra_op_parallelism_threads,
            deterministic_ops=self.deterministic_ops,
        )

    @classmethod
    def from_config(cls, config):
        return TfConfig(**config)
