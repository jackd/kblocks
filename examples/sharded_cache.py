from typing import Dict
import tensorflow as tf
import tensorflow_datasets as tfds


class CacheConfig(tfds.core.BuilderConfig):

    def __init__(self,
                 name,
                 datasets: Dict[str, tf.data.Dataset],
                 description='cached dataset'):
        self._datasets = datasets
        first, *rest = (ds.example_spec for ds in tf.nest.flatten(datasets))
        for r in rest:
            if not tf.nest.assert_same_structure(first, r):
                raise ValueError('dataset element_specs not same structure')
            struct = tf.nest.map_structure(lambda a, b: a == b, first, r)
            if not all(tf.nest.flatten(struct)):
                raise ValueError('dataset element_specs not all the same')
        self._example_spec = first

        super().__init__(name=name, version='0.0.1', description=description)

    @property
    def example_spec(self):
        return self._example_spec

    @property
    def flat_features(self):
        return tuple(
            tfds.core.features.Tensor(shape=x.shape, dtype=x.dtype)
            for x in tf.nest.flatten(self.example_spec, expand_composites=True))

    def repack(self, example):
        return tf.nest.pack_sequence_as(self.example_spec,
                                        example,
                                        expand_composites=True)

    @property
    def datasets(self):
        return self._datasets.copy()  # defensive copy


def _as_feature_dict(example):
    return {f'feature-{i}': v for i, v in enumerate(example)}


class TfdsCache(tfds.core.GeneratorBasedBuilder):

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.core.features.FeaturesDict(
                _as_feature_dict(self.builder_config.flat_features)))

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            tfds.core.SplitGenerator(name=split,
                                     gen_kwargs=dict(dataset=dataset))
            for split, dataset in self.builder_config.datasets.items()
        ]

    def _generate_examples(self, dataset):
        """Generate NMNIST examples as dicts."""
        for i, example in dataset:
            yield i, _as_feature_dict(example)
