# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import tensorflow as tf

# class OptimizerScope(object):

#     def __init__(self, optimizer: tf.keras.optimizers.Optimizer):
#         self._optimizer = optimizer

#     _stack = []

#     def __enter__(self):
#         OptimizerScope._stack.append(self)

#     def __exit__(self, type, value, traceback):
#         out = OptimizerScope._stack.pop()
#         assert (out is self)

#     @classmethod
#     def get_default(cls) -> 'OptimizerScope':
#         if len(cls._stack) == 0:
#             raise RuntimeError('Cannot `get_default` {} - stack empty'.format(
#                 cls.__name__))
#         return cls._stack[-1]

#     @property
#     def optimizer(self) -> tf.keras.optimizers.Optimizer:
#         return self._optimizer

# def get_iterations():
#     return get_default().optimizer.iterations

# def get_default() -> OptimizerScope:
#     return OptimizerScope.get_default()
