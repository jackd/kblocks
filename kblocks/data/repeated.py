from typing import Optional, Tuple, Union

import gin
import tensorflow as tf


@gin.configurable(module="kb.data")
class RepeatedData:
    """
    Immutable class for encompassing an infinite dataset with specified steps_per_epoch.

    By construction, the `dataset` attribute will have infinite cardinality.

    This is helpful for making pre-emptible, deterministic data pipelines. For example,
    a pipeline that involves a post-batch map with random elements might be implemented
    as follows.

    ```python
    import tfrng
    def sample_map_func(inputs, labels):
        return inputs + tfrng.normal(shape=tf.shape(inputs)), labels

    def get_train_data(base_dataset, map_func, seed):
        steps_per_epoch = len(dataset)
        dataset = dataset.repeat().apply(tfrng.data.stateless_map(map_func, seed=seed)
        return DatasetWithLength(dataset, steps_per_epoch)
    ```

    This will give data augmentation that is different each epoch.
    """

    def __init__(self, dataset: tf.data.Dataset, steps_per_epoch: Optional[int] = None):
        cardinality = tf.keras.backend.get_value(dataset.cardinality())
        if steps_per_epoch is None:
            steps_per_epoch = cardinality
            if cardinality == tf.data.INFINITE_CARDINALITY:
                raise ValueError(
                    "steps_per_epoch must be provided if dataset has infinite "
                    "cardinality"
                )
            dataset = dataset.repeat()
        elif cardinality != tf.data.INFINITE_CARDINALITY:
            assert cardinality == steps_per_epoch
            dataset = dataset.repeat()
        self._dataset = dataset
        self._steps_per_epoch = steps_per_epoch

    @property
    def steps_per_epoch(self) -> int:
        return self._steps_per_epoch

    @property
    def dataset(self) -> tf.data.Dataset:
        return self._dataset


@gin.register(module="kb.data")
def repeated_data(data: Union[tf.data.Dataset, RepeatedData]) -> RepeatedData:
    if isinstance(data, tf.data.Dataset):
        return RepeatedData(data)
    if isinstance(data, RepeatedData):
        return data
    raise TypeError(f"data must be a Dataset or RepeatedData, got {data}")


def dataset_and_steps(
    data: Union[tf.data.Dataset, RepeatedData], steps: Optional[int] = None
) -> Tuple[tf.data.Dataset, Optional[int]]:
    """Get consistent (dataset, steps_per_epoch)."""
    if isinstance(data, RepeatedData):
        assert steps is None or steps == data.steps_per_epoch
        return data.dataset, data.steps_per_epoch
    assert isinstance(data, tf.data.Dataset)
    cardinality = tf.keras.backend.get_value(data.cardinality())
    if steps is None:
        assert cardinality > 0
        return data, None
    if cardinality == tf.data.INFINITE_CARDINALITY:
        return data, steps
    if cardinality != steps:
        raise ValueError(
            "`steps` and `data.cardinality()` must be consistent, "
            f"but {steps} != {cardinality}"
        )
    return data, None
