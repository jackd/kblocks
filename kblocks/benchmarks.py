"""Utilities for reporting results from `tf.test.Benchmark.run_op_benchmark`."""
import gin
import tensorflow as tf

from kblocks.data.sources import DataSource
from kblocks.trainables import Trainable


def summarize(result, print_fn=print):
    """
    Args:
        result: output of a tf.test.Benchmark.run_op_benchmark call.
        print_fn: print-like function.
    """
    print_fn("Wall time (ms): {}".format(result["wall_time"] * 1000))
    gpu_mem = result["extras"].get("allocator_maximum_num_bytes_GPU_0_bfc", 0)
    print_fn("Memory (Mb):    {}".format(gpu_mem / 1024 ** 2))


def summarize_all(*args, print_fn=print):
    """
    Applies `summarize` to (name, result) pairs.

    Args:
        *args: (name, result) pairs
        print_fn: print-like function.
    """
    for name, result in args:
        print_fn(name)
        summarize(result, print_fn)


@gin.configurable(module="kb.benchmarks")
def benchmark_op(op, burn_iters: int = 2, min_iters: int = 10):
    """Final endpoint for all kb.benchmarks functions."""
    assert not tf.executing_eagerly()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        bm = tf.test.Benchmark()
        result = bm.run_op_benchmark(
            sess, op, burn_iters=burn_iters, min_iters=min_iters
        )
        summarize(result)
    return result


def as_iterator(dataset):
    return tf.compat.v1.data.make_one_shot_iterator(dataset)


def as_inputs(dataset):
    return as_iterator(dataset).get_next()


@gin.configurable(module="kb.benchmarks")
def benchmark_dataset(dataset: tf.data.Dataset, **kwargs):
    return benchmark_op(as_inputs(dataset.repeat()), **kwargs)


@gin.configurable(module="kb.benchmarks")
def benchmark_model(
    model: tf.keras.Model, dataset: tf.data.Dataset, inference_only=False, **kwargs
):
    inputs, labels, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(
        as_inputs(dataset.repeat())
    )
    if inference_only:
        op = model(inputs)
    else:
        variables = model.trainable_variables
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = model.loss(labels, predictions, sample_weight=sample_weight)
        grads = tape.gradient(loss, variables)
        op = model.optimizer.apply_gradients(zip(grads, variables))
    return benchmark_op(op, **kwargs)

    # func = (
    #     model.make_predict_function() if inference_only else
    #     model.make_train_function()
    # )
    # op = func(tf.compat.v1.data.make_one_shot_iterator(dataset.repeat()))
    # return benchmark_op(op, **kwargs)


@gin.configurable(module="kb.benchmarks")
def benchmark_trainable(
    trainable: Trainable, train_split=True, source_only=False, **kwargs
):
    source = trainable.train_source if train_split else trainable.validation_source
    dataset = source.dataset
    if source_only:
        return benchmark_dataset(dataset, **kwargs)
    return benchmark_model(trainable.model, dataset, **kwargs)


@gin.configurable(module="kb.benchmarks")
def benchmark_source(source: DataSource, **kwargs):
    return benchmark_dataset(source.dataset, **kwargs)
