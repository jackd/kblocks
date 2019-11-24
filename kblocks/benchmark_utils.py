"""Utilities for reporting results from `tf.test.Benchmark.run_op_benchmark`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def summarize(result, print_fn=print):
    """
    Args:
        result: output of a tf.test.Benchmark.run_op_benchmark call.
        print_fn: print-like function.
    """
    print_fn('Wall time (ms): {}'.format(result['wall_time'] * 1000))
    print_fn('Memory (Mb):    {}'.format(
        result['extras']['allocator_maximum_num_bytes_GPU_0_bfc'] / 1024**2))


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
