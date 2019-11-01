from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
# from kblocks.gin_utils import configurable
import gin


@gin.configurable()
def f(x: int, y: int) -> int:
    return x + y


# this should create an error in vscode
# print(f('hello', 'world'))
# fixed
print(f(2, 3))


class Base(object):

    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b

    def config(self) -> Tuple[int, int]:
        return (self.a, self.b)


@gin.configurable()
class A(Base):

    def __init__(self, a: int, b: int):
        super(A, self).__init__(a + 1, b + 1)

    def config(self) -> Tuple[int, int]:
        return (self.a, self.b)


# this should create an error in vscode
# A('2', '3')
# fixed
A(2, 3)
