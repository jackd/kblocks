from absl import logging
from typing import Callable, Dict, Optional, Iterable, Any, Tuple
import collections

import tensorflow as tf
import tensorflow_datasets as tfds

import gin

from kblocks.framework.sources import DataSource
from kblocks.framework.sources import TfdsSource


class CacheConfig(tfds.core.BuilderConfig):
    def __init__(
        self, name, datasets: Dict[str, tf.data.Dataset], description="cached dataset"
    ):
        self._datasets = datasets
        first, *rest = (ds.element_spec for ds in tf.nest.flatten(datasets))
        for r in rest:
            tf.nest.assert_same_structure(first, r)
            struct = tf.nest.map_structure(lambda a, b: a == b, first, r)
            if not all(tf.nest.flatten(struct)):
                lines = ["dataset element_specs not all the same"]
                for i, s in enumerate((first, *rest)):
                    lines.append(f"Spec {i}: {s}")
                raise ValueError("\n".join(lines))
        self._example_spec = first

        super().__init__(name=name, version="0.0.1", description=description)

    @property
    def example_spec(self):
        return self._example_spec

    @property
    def datasets(self):
        return self._datasets.copy()  # defensive copy


def spec_to_feature(spec):
    assert spec.shape.ndims is not None
    if isinstance(spec, tf.TensorSpec):
        return tfds.core.features.Tensor(shape=spec.shape, dtype=spec.dtype)
    elif isinstance(spec, (tf.RaggedTensorSpec, tf.SparseTensorSpec)):
        raise NotImplementedError()
    else:
        raise TypeError(f"Unrecognized spec type {spec}")


class TfdsCache(tfds.core.GeneratorBasedBuilder):
    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.core.features.FeaturesDict(
                tf.nest.map_structure(spec_to_feature, self.builder_config.example_spec)
            ),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            tfds.core.SplitGenerator(name=split, gen_kwargs=dict(dataset=dataset))
            for split, dataset in self.builder_config.datasets.items()
        ]

    def _generate_examples(self, dataset):
        """Generate NMNIST examples as dicts."""
        for i, example in enumerate(dataset):
            yield (i, example)


@gin.configurable(module="kb.framework")
def cached_source(
    name: str,
    base_source: DataSource,
    pre_cache_map: Callable,
    cache_repeats: Optional[int] = None,
    splits: Iterable[str] = ("train", "validation"),
    meta=None,
    **kwargs,
):
    """Assumes pre_cache_map returns a dict."""
    datasets = {}

    for split in splits:
        training = split == "train"
        with tf.keras.backend.learning_phase_scope(training):
            ds = base_source.get_dataset(split).map(pre_cache_map)
            spec = ds.element_spec
            for s in tf.nest.flatten(spec):
                if s.shape.ndims is None:
                    raise ValueError("All specs must have number of dimensions known.")
            if not isinstance(spec, collections.Mapping):
                raise ValueError(
                    f"map function must return a dict of tensors, " "got spec {spec}"
                )
            if cache_repeats is not None and training:
                assert cache_repeats != -1
                ds = ds.repeat(cache_repeats)

            datasets[split] = ds
    config = CacheConfig(name, datasets)
    builder = TfdsCache(config=config)
    if meta is None:
        meta = base_source.meta
    return TfdsSource(builder=builder, as_supervised=False, meta=meta, **kwargs)
