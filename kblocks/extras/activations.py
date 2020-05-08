import gin
from kblocks.ops import complex as complex_ops

swish = gin.external_configurable(complex_ops.swish, module="kb.extras.activations")
modified_swish = gin.external_configurable(
    complex_ops.modified_swish, module="kb.extras.activations"
)
softplus = gin.external_configurable(
    complex_ops.softplus, module="kb.extras.activations"
)

complex_relu = gin.external_configurable(
    complex_ops.complex_relu, module="kb.extras.activations"
)
