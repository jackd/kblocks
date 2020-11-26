import abc
from typing import Any, Callable, Dict, Iterable, Optional

import gin
import tensorflow as tf
import tqdm

import kblocks.data.transforms.tfrecords as tfrecords_lib
from kblocks.data.transforms.core import Transform, _get_rng, _maybe_ragged_batch
from kblocks.serialize import register_serializable

if tf.version.VERSION < "2.4":

    def _dataset_from_generator(generator: Callable[[], Iterable], output_signature):
        if all(
            isinstance(spec, tf.TensorSpec)
            for spec in tf.nest.flatten(output_signature)
        ):
            # no composites
            return tf.data.Dataset.from_generator(
                generator,
                output_shapes=tf.nest.map_structure(
                    lambda spec: spec.shape, output_signature
                ),
                output_types=tf.nest.map_structure(
                    lambda spec: spec.dtype, output_signature
                ),
            )

        # composites - flatten in generator, repack in map
        def actual_gen():
            for element in generator():
                yield tuple(tf.nest.flatten(element, expand_composites=True))

        spec = tf.nest.flatten(output_signature, expand_composites=True)
        dataset = tf.data.Dataset.from_generator(
            actual_gen,
            output_shapes=tuple(s.shape for s in spec),
            output_types=tuple(s.dtype for s in spec),
        )
        return dataset.map(
            lambda *args: tf.nest.pack_sequence_as(
                output_signature, args, expand_composites=True
            )
        )


else:

    def _dataset_from_generator(generator: Callable[[], Iterable], output_signature):
        return tf.data.Dataset.from_generator(
            generator, output_signature=output_signature
        )


class ShuffleRng(Transform, abc.ABC):
    def __init__(self, buffer_size: int, seed: Optional[int] = None):
        self._buffer_size = buffer_size
        self._seed = seed
        self._rng = None
        super().__init__()

    def get_config(self) -> Dict[str, Any]:
        return dict(buffer_size=self._buffer_size, seed=self._seed)


@gin.configurable(module="kb.data")
@register_serializable
class ShuffleScan(ShuffleRng):
    """
    Shuffle transform that uses a `tf.random.Generator` with external state.

    This implementation uses `tf.data.experimental.scan` to achieve same functionality
    as `tf.data.Dataset.shuffle`, though may be exceedingly slow.
    """

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        @tf.function
        def scan_func(old_state, input_element):
            if self._rng is None:
                self._rng = _get_rng(self._seed)

            state_index, counter = old_state
            valid, example_string = input_element

            if state_index == 0:
                # fill buffer
                buffer[counter].assign(example_string)
                counter += 1
                state_index = 0 if counter < buffer_size else 1
                output_element = (False, example_string)
                new_state = (state_index, counter)
                return new_state, output_element

            def scan_state2():
                # Exhaust buffer when main dataset exhausted
                new_state = (state_index, counter + 1)
                output_element = (True, buffer[counter])
                return new_state, output_element

            if state_index == 1:
                if valid:
                    # sample / replace
                    i = self._rng.uniform((), maxval=buffer_size, dtype=tf.int64)
                    output_element = (True, buffer[i])
                    buffer[i].assign(example_string)
                    new_state = (state_index, counter)
                    return new_state, output_element

                # main dataset exhausted - elements are now just padding
                # trainsition to exhausting the buffer without replacement
                state_index = 2
                # shuffle buffer using rng
                i = self._rng.uniform((buffer_size,), dtype=tf.float32)
                perm = tf.argsort(i)
                buffer.assign(tf.gather(buffer, perm, axis=0))
                counter = 0

                # tf.function needs return statement in this branch
                return scan_state2()

            tf.assert_equal(state_index, 2)
            return scan_state2()

        cardinality = dataset.cardinality()
        buffer_size = self._buffer_size
        spec = dataset.element_spec
        deserialize = tfrecords_lib.deserializer(spec)

        dataset = dataset.map(
            lambda *args: (True, tfrecords_lib.serialize_example(*args))
        )
        padding = tf.data.Dataset.from_tensors(
            (False, tf.zeros((), dtype=tf.string))
        ).repeat(buffer_size)
        dataset = dataset.concatenate(padding)

        buffer = tf.Variable(tf.zeros((buffer_size,), dtype=tf.string), dtype=tf.string)

        initial_state = (0, 0)

        dataset = dataset.apply(tf.data.experimental.scan(initial_state, scan_func))
        dataset = dataset.filter(lambda valid, example_string: valid)
        dataset = dataset.map(lambda valid, example_string: deserialize(example_string))
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(cardinality))
        return dataset


@gin.configurable(module="kb.data")
@register_serializable
class ShuffleGen(ShuffleRng):
    """
    Shuffle implementation that maintains random state implemented with numpy generator.

    This implementation should be equivalent to `tf.data.Dataset.shuffle` though is
    considerably slower.
    """

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        def gen():
            if self._rng is None:
                self._rng = _get_rng(self._seed)
            iterator = iter(dataset)
            buffer = []

            try:
                for _ in tqdm.trange(
                    self._buffer_size, desc="Filling shuffler buffer..."
                ):
                    buffer.append(next(iterator))
                size = len(buffer)
                while True:
                    next_value = next(iterator)
                    i = self._rng.uniform((), maxval=size, dtype=tf.int64)
                    yield buffer[i]
                    buffer[i] = next_value
            except StopIteration:
                # exhaust buffer
                u = self._rng.uniform((len(buffer),))
                perm = tf.argsort(u)
                for p in perm:
                    yield buffer[p]

        shuffled = _dataset_from_generator(gen, dataset.element_spec)
        return shuffled.apply(
            tf.data.experimental.assert_cardinality(dataset.cardinality())
        )


@gin.configurable(module="kb.data")
@register_serializable
class ShuffleBatch(ShuffleRng):
    """
    Shuffle implementation that maintains random state implemented with numpy generator.

    This implementation uses a batch, shuffle and unbatch transformation. This is not
    equivalent to that used by `tf.data.Dataset.shuffle` which maintains a buffer,
    but should be relatively fast compared to other stateful shuffle implementations.
    """

    def __call__(self, dataset: tf.data.Dataset):
        def map_func(*args):
            if len(args) == 1:
                (args,) = args
            if self._rng is None:
                self._rng = _get_rng()
            el = tf.nest.flatten(args)[0]
            if isinstance(el, tf.RaggedTensor):
                batch_size = el.nrows()
            elif isinstance(el, tf.SparseTensor):
                batch_size = el.dense_shape[0]
            elif isinstance(el, tf.Tensor):
                batch_size = tf.shape(el)[0]
            else:
                raise TypeError(f"Invalid leading element {type(el)}")
            i = self._rng.uniform(((batch_size,)))
            perm = tf.argsort(i)
            return tf.nest.map_structure(lambda x: tf.gather(x, perm, axis=0), args)

        return (
            _maybe_ragged_batch(dataset, batch_size=self._buffer_size)
            .map(map_func)
            .unbatch()
        )
