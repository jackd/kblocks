from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import gin
import tensorflow as tf

from kblocks.data.sources import DataSource
from kblocks.data.sources import get as get_source
from kblocks.extras import callbacks as cb
from kblocks.keras import model as model_lib
from kblocks.serialize import register_serializable


@gin.configurable(module="kb.trainable")
@register_serializable
class Trainable(tf.Module):
    """
    A `tf.keras.Model`'s and additional objects making up a training state.

    A model's weights make up a good deal of the training state. However, in order to
    achieve truly reproducible / pre-emptible training, there are other components of
    training that must be saved / restored. These include the state of data pipelines
    (e.g. random state for data augmentation and shuffling) and some callbacks (e.g.
    `EarlyStopping`, `ReduceLROnPlateau`).
    """

    def __init__(
        self,
        model: tf.keras.Model,
        train_source: DataSource,
        validation_source: Optional[DataSource] = None,
        callbacks: Sequence[tf.keras.callbacks.Callback] = (),
        name: Optional[str] = None,
    ):
        model = model_lib.get(model)
        train_source = get_source(train_source)
        if validation_source is not None:
            validation_source = get_source(validation_source)
        callbacks = tuple((cb.get(c) for c in callbacks))
        model_lib.assert_compiled(model)
        model_lib.init_optimizer_weights(model)

        model_lib.assert_valid_cardinality(train_source.dataset.cardinality())

        self._model = model
        self._train_source = train_source
        self._validation_source = validation_source
        self._callbacks = callbacks

        self._model._train_source = train_source
        if validation_source is not None:
            self._model._validation_source = validation_source
        super().__init__(name=name)

    def get_config(self):
        return tf.nest.map_structure(
            tf.keras.utils.serialize_keras_object,
            dict(
                model=self.model,
                train_source=self.train_source,
                validation_source=self.validation_source,
                callbacks=self.callbacks,
            ),
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    @property
    def train_source(self) -> DataSource:
        return self._train_source

    @property
    def validation_source(self) -> Optional[DataSource]:
        return self._validation_source

    @property
    def callbacks(self) -> Tuple[tf.keras.callbacks.Callback]:
        return self._callbacks


@gin.configurable(module="kb.trainable")
def build_trainable(
    model_fn: Callable[[Any], tf.keras.Model],
    train_source: DataSource,
    validation_source: Optional[DataSource] = None,
    callbacks: Iterable[tf.keras.callbacks.Callback] = (),
    loss=None,
    metrics=None,
    optimizer=None,
):
    model = model_fn(train_source.dataset.element_spec[0])
    if optimizer is None:
        assert loss is None
        assert metrics is None
    else:
        model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return Trainable(
        model=model,
        train_source=train_source,
        validation_source=validation_source,
        callbacks=callbacks,
    )


def get(identifier) -> Trainable:
    if isinstance(identifier, Trainable):
        return identifier

    trainable = tf.keras.utils.deserialize_keras_object(identifier)
    if not isinstance(trainable, Trainable):
        raise ValueError(f"Invalid trainable: {trainable}")
    return trainable
