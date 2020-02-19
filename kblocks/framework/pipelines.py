import os
import abc
from absl import logging
from typing import Callable, Optional
import tensorflow as tf
import gin
from tqdm import tqdm

AUTOTUNE = tf.data.experimental.AUTOTUNE


def _always_throws():
    raise NotImplementedError('Should not be called')


class DataPipeline(abc.ABC):

    def processed_spec(self, example_spec):
        dataset = tf.data.Dataset.from_generator(
            _always_throws,
            output_types=tf.nest.map_structure(lambda x: x.dtype, example_spec),
            output_shapes=tf.nest.map_structure(lambda x: x.shape,
                                                example_spec))
        return self(dataset, 'train').element_spec

    @abc.abstractproperty
    def batch_size(self) -> int:
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def __call__(self, dataset: tf.data.Dataset, split: str) -> tf.data.Dataset:
        raise NotImplementedError('Abstract method')


@gin.configurable(module='kb.framework')
class BasePipeline(DataPipeline):

    def __init__(self,
                 batch_size: int,
                 pre_cache_map: Optional[Callable] = None,
                 pre_batch_map: Optional[Callable] = None,
                 post_batch_map: Optional[Callable] = None,
                 cache_dir: Optional[str] = None,
                 shuffle_buffer: Optional[int] = None,
                 repeats: Optional[int] = None,
                 prefetch_buffer: int = AUTOTUNE,
                 num_parallel_calls: int = AUTOTUNE,
                 cache_repeats: Optional[int] = None,
                 clear_cache: bool = False,
                 pre_cache: bool = True,
                 drop_remainder: bool = True):
        if cache_dir is not None:
            cache_dir = os.path.expandvars(os.path.expanduser(cache_dir))
        self._pre_cache_map = pre_cache_map
        self._pre_batch_map = pre_batch_map
        self._post_batch_map = post_batch_map

        self._batch_size = batch_size
        self._cache_dir = cache_dir
        self._shuffle_buffer = shuffle_buffer
        self._repeats = repeats
        self._prefetch_buffer = prefetch_buffer
        self._num_parallel_calls = num_parallel_calls
        self._cache_repeats = cache_repeats
        self._pre_cache = pre_cache
        self._drop_remainder = drop_remainder
        if clear_cache:
            self.clear_cache()

    def clear_cache(self):
        cache_dir = self._cache_dir
        if cache_dir is None:
            logging.info(
                'Tried to clear cache, but no cache_dir specified - ignoring')
            return
        if not os.path.isdir(cache_dir):
            logging.info(
                'Tried to clear cache, but cache_dir does not exist - ignoring')
            return

        logging.info('Clearing cache at {}'.format(cache_dir))
        tf.io.gfile.rmtree(cache_dir)

    # def get_config(self):
    #     return dict(pre_cache_map=self._pre_cache_map,
    #                 pre_batch_map=self._post_batch_map,
    #                 post_batch_map=self._post_batch_map,
    #                 batch_size=self._batch_size,
    #                 cache_dir=self._cache_dir,
    #                 shuffle_buffer=self._shuffle_buffer,
    #                 repeats=self._repeats,
    #                 prefetch_buffer=self._prefetch_buffer,
    #                 num_parallel_calls=self._num_parallel_calls,
    #                 cache_repeats=self._cache_repeats,
    #                 pre_cache=self._pre_cache,
    #                 drop_remainder=self._drop_remainder)

    def cache_path(self, split) -> str:
        if self._cache_dir is None:
            return ''
        return os.path.join(self._cache_dir, 'cache', split)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def processed_spec(self, example_spec):
        dataset = tf.data.Dataset.from_generator(
            _always_throws,
            output_types=tf.nest.map_structure(lambda x: x.dtype, example_spec),
            output_shapes=tf.nest.map_structure(lambda x: x.shape,
                                                example_spec))
        return self._process_dataset(dataset,
                                     split='train',
                                     preprocess_cache=False).element_spec

    def __call__(self, dataset: tf.data.Dataset, split: str):
        return self._process_dataset(dataset, split)

    def _cache(self, dataset, cache_path, preprocess_cache):
        training = tf.keras.backend.learning_phase()
        if self._cache_repeats is not None:
            if training:
                dataset = dataset.repeat(self._cache_repeats)
        if cache_path != '':
            root_dir = os.path.dirname(cache_path)
            if root_dir is not None and not os.path.isdir(root_dir):
                os.makedirs(root_dir)
        dataset = dataset.cache(cache_path)
        index_path = '{}.index'.format(cache_path)
        if not os.path.isfile(index_path) and preprocess_cache:
            if cache_path == '':
                logging.info('Creating cache in memory')
            else:
                logging.info('Creating cache at {}'.format(cache_path))
            if tf.executing_eagerly():
                for example in tqdm(dataset, desc='Caching (eager)...'):
                    del example
            else:
                example = tf.compat.v1.data.make_one_shot_iterator(
                    dataset).get_next()
                with tf.compat.v1.Session() as sess:
                    try:
                        pbar = tqdm(desc='Caching (session)...')
                        while True:
                            sess.run(example)
                            pbar.update()
                    except tf.errors.OutOfRangeError:
                        pass
        return dataset

    def _process_dataset(self,
                         dataset: tf.data.Dataset,
                         split: str,
                         preprocess_cache=None) -> tf.data.Dataset:
        with tf.keras.backend.learning_phase_scope(split == 'train'):
            if preprocess_cache is None:
                preprocess_cache = self._pre_cache
            if self._pre_cache_map is not None:
                dataset = dataset.map(self._pre_cache_map,
                                      self._num_parallel_calls)
                dataset = self._cache(dataset, self.cache_path(split),
                                      preprocess_cache)

            if self._repeats != -1:
                dataset = dataset.repeat(self._repeats)

            if self._shuffle_buffer is not None:
                dataset = dataset.shuffle(self._shuffle_buffer)

            if self._pre_batch_map is not None:
                dataset = dataset.map(self._pre_batch_map,
                                      self._num_parallel_calls)

            dataset = dataset.batch(self.batch_size,
                                    drop_remainder=self._drop_remainder)

            if self._post_batch_map is not None:
                dataset = dataset.map(self._post_batch_map,
                                      self._num_parallel_calls)

            if self._prefetch_buffer is not None:
                dataset = dataset.prefetch(self._prefetch_buffer)
            return dataset
