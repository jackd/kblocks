import abc
from typing import Optional

import gin
import tensorflow as tf


class Batcher(abc.ABC):
    @abc.abstractmethod
    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        raise NotImplementedError("Abstract method")

    def epoch_length(  # pylint:disable=no-self-use
        self, example_epoch_length: Optional[int]
    ) -> Optional[int]:
        # default returns None
        del example_epoch_length


@gin.configurable(module="kb.framework")
class RectBatcher(Batcher):
    def __init__(self, batch_size: int, drop_remainder: bool = False):
        assert isinstance(batch_size, int)
        self._batch_size = batch_size
        self._drop_remainder = drop_remainder

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.batch(self._batch_size, self._drop_remainder)

    def epoch_length(self, example_epoch_length: Optional[int]) -> Optional[int]:
        if example_epoch_length is None:
            return None
        epoch_length = example_epoch_length // self._batch_size
        if self._drop_remainder or example_epoch_length % self._batch_size == 0:
            return epoch_length
        return epoch_length + 1


@gin.configurable(module="kb.framework")
class RaggedBatcher(RectBatcher):
    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(
                self._batch_size, self._drop_remainder
            )
        )
