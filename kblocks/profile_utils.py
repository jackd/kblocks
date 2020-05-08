from __future__ import absolute_import, division, print_function

import itertools
from typing import Any, Callable, Iterable

import numpy as np
import tensorflow as tf

MultiGraphNodeProto = tf.compat.v1.profiler.MultiGraphNodeProto
GraphNodeProto = tf.compat.v1.profiler.GraphNodeProto


def _extract_nodes_recursive(
    multi_node: MultiGraphNodeProto,
) -> Iterable[GraphNodeProto]:
    yield multi_node.graph_nodes
    for child in multi_node.children:
        yield from _extract_nodes_recursive(child)


def extract_nodes(multi_node: MultiGraphNodeProto) -> Iterable[GraphNodeProto]:
    return itertools.chain(*_extract_nodes_recursive(multi_node))


def extract_multi_nodes(
    multi_node: MultiGraphNodeProto,
) -> Iterable[MultiGraphNodeProto]:
    yield multi_node
    for child in multi_node.children:
        yield from extract_multi_nodes(child)


def total_peak_bytes(multi_node: MultiGraphNodeProto) -> int:
    return sum(n.peak_bytes for n in extract_nodes(multi_node))


def total_requested_bytes(multi_node: MultiGraphNodeProto) -> int:
    return multi_node.total_requested_bytes


def sum_residual_bytes(multi_node: MultiGraphNodeProto) -> int:
    return sum(n.residual_bytes for n in extract_nodes(multi_node))


def summarize(
    profiles: Iterable[MultiGraphNodeProto], print_fn: Callable[[Any], Any] = print
):
    peak_bytes = np.array([total_peak_bytes(p) for p in profiles]) / 1024 ** 2
    req_bytes = np.array([total_requested_bytes(p) for p in profiles]) / 1024 ** 2
    msg = "{:25} : {}"
    print_fn(msg.format("max_peak_bytes (Mb)", np.max(peak_bytes)))
    print_fn(msg.format("max_req_bytes (Mb)", np.max(req_bytes)))
    print_fn(msg.format("mean_peak_bytes (Mb)", np.mean(peak_bytes)))
    print_fn(msg.format("mean_req_bytes (Mb)", np.mean(req_bytes)))

    for k in (
        "total_exec_micros",
        "total_requested_bytes",
        "total_peak_bytes",
        "total_residual_bytes",
        "total_output_bytes",
    ):
        print_fn(msg.format(k, np.mean([getattr(p, k) for p in profiles])))
    print_fn(
        msg.format(
            "sum_residual_bytes", np.mean([sum_residual_bytes(p) for p in profiles])
        )
    )
