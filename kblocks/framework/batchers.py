import gin

from meta_model import batchers

Batcher = gin.configurable(batchers.Batcher, module="kb.framework")
RectBatcher = gin.configurable(batchers.RectBatcher, module="kb.framework")
RaggedBatcher = gin.configurable(batchers.RaggedBatcher, module="kb.framework")
RaggedBatcherV2 = gin.configurable(batchers.RaggedBatcherV2, module="kb.framework")
PaddedRaggedBatcher = gin.configurable(
    batchers.PaddedRaggedBatcher, module="kb.framework"
)
