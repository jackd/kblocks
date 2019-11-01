from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import itertools
import six
import numpy as np
import tensorflow as tf
import gin

from typing import Optional, Sequence, Union, Tuple

Num = Union[int, float]


def factorial(n: int) -> int:
    return np.prod(range(1, n + 1))


def get_geometric_polynomials(x: tf.Tensor, order: int) -> tf.Tensor:
    orders = tf.range(order, dtype=tf.float32)
    return tf.expand_dims(x, axis=-1)**orders


class PolynomialBuilder(abc.ABC):

    @abc.abstractmethod
    def get_polynomials(self, x: tf.Tensor, order: int) -> Sequence[tf.Tensor]:
        raise NotImplementedError

    def __call__(self, x: tf.Tensor, order: int) -> Sequence[tf.Tensor]:
        return self.get_polynomials(x, order)


@gin.configurable(module='kblocks.ops')
class GeometricPolynomialBuilder(PolynomialBuilder):

    def get_polynomials(self, x: tf.Tensor, order: int) -> Sequence[tf.Tensor]:
        return tf.unstack(get_geometric_polynomials(x, order), axis=-1)

    def __repr__(self):
        return 'GeomPolyBuilder'


class OrthogonalPolynomialBuilder(PolynomialBuilder):

    @abc.abstractmethod
    def get_polynomials(self, x: tf.Tensor, order: int) -> Sequence[tf.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_domain(self):
        raise NotImplementedError

    def get_normalization_factor(self, order: int) -> Num:
        return 1

    def get_weighting_fn(self, x: tf.Tensor) -> tf.Tensor:
        return tf.ones_like(x)


class RecursiveOrthogonalPolynomialBuilder(OrthogonalPolynomialBuilder):

    def get_p0(self, x: tf.Tensor) -> tf.Tensor:
        return tf.ones_like(x)

    def get_p1(self, x: tf.Tensor) -> tf.Tensor:
        return x

    @abc.abstractmethod
    def get_next(self, x: tf.Tensor, pn2: tf.Tensor, pn1: tf.Tensor,
                 n: int) -> tf.Tensor:
        raise NotImplementedError

    def get_polynomials(self, x: tf.Tensor, order: int) -> tf.Tensor:
        if order < 0:
            raise ValueError('Order must be non-negative')
        p0 = self.get_p0(x)
        if order == 1:
            return [p0]
        p1 = self.get_p1(x)
        ps = [p0, p1]
        for n in range(2, order):
            ps.append(self.get_next(x, ps[n - 2], ps[n - 1], n))
        return ps


@gin.configurable(module='kblocks.ops')
class LegendrePolynomialBuilder(RecursiveOrthogonalPolynomialBuilder):

    def get_next(self, x: tf.Tensor, pn2: tf.Tensor, pn1: tf.Tensor,
                 n: int) -> tf.Tensor:
        n -= 1
        return (2 * n + 1) / (n + 1) * x * pn1 - n / (n + 1) * pn2

    def get_normalization_factor(self, order: int) -> float:
        return 2 / (2 * order + 1)

    def get_domain(self) -> Tuple[Num, Num]:
        return (-1, 1)

    def __repr__(self):
        return 'LegendrePolyBuilder'


class ChebyshevPolynomialBuilder(RecursiveOrthogonalPolynomialBuilder):

    def get_next(self, x: tf.Tensor, pn2: tf.Tensor, pn1: tf.Tensor,
                 n: int) -> tf.Tensor:
        return 2 * x * pn1 - pn2

    def get_domain(self) -> Tuple[Num, Num]:
        return (-1, 1)

    @staticmethod
    def from_kind(kind='first'):
        if kind == 'first':
            return FirstChebyshevPolynomialBuilder()
        elif kind == 'second':
            return SecondChebyshevPolynomialBuilder()
        else:
            raise ValueError('`kind` must be one of "first", "second"')


@gin.configurable(module='kblocks.ops')
class FirstChebyshevPolynomialBuilder(ChebyshevPolynomialBuilder):

    def get_p1(self, x: tf.Tensor) -> tf.Tensor:
        return x

    def get_weighting_fn(self, x: tf.Tensor) -> tf.Tensor:
        return 1 / tf.sqrt(1 - x**2)

    def get_normalization_factor(self, order: int) -> Num:
        return np.pi if order == 0 else np.pi / 2

    def __repr__(self):
        return 'Chebyshev1PolyBuilder'


@gin.configurable(module='kblocks.ops')
class SecondChebyshevPolynomialBuilder(ChebyshevPolynomialBuilder):

    def get_p1(self, x: tf.Tensor) -> tf.Tensor:
        return 2 * x

    def get_weighting_fn(self, x: tf.Tensor) -> tf.Tensor:
        return tf.sqrt(1 - tf.square(x))

    def get_normalization_factor(self, order: int) -> float:
        return np.pi / 2

    def __repr__(self):
        return 'Chebyshev2PolyBuilder'


@gin.configurable(module='kblocks.ops')
class HermitePolynomialBuilder(RecursiveOrthogonalPolynomialBuilder):

    def get_p1(self, x: tf.Tensor) -> tf.Tensor:
        return 2 * x

    def get_next(self, x: tf.Tensor, pn2: tf.Tensor, pn1: tf.Tensor,
                 n: int) -> tf.Tensor:
        return 2 * x * pn1 - 2 * (n - 1) * pn2

    def get_domain(self) -> Tuple[float, float]:
        return (-np.inf, np.inf)

    def get_weighting_fn(self, x: tf.Tensor) -> tf.Tensor:
        return tf.exp(-tf.square(x))

    def get_normalization_factor(self, order):
        return factorial(order) * 2**order * np.sqrt(np.pi)

    def __repr__(self):
        return 'HermitePolyBuilder'


@gin.configurable(module='kblocks.ops')
class GaussianHermitePolynomialBuilder(OrthogonalPolynomialBuilder):

    def __init__(self, stddev: Union[Num, tf.Tensor, tf.Variable] = 1.0):
        self.stddev = stddev

    def get_polynomials(self, x: tf.Tensor, order: int) -> tf.Tensor:
        hermites = HermitePolynomialBuilder().get_polynomials(
            x / self.stddev, order)
        f = np.sqrt(np.pi) * self.stddev
        exp_denom = 2 * tf.square(self.stddev)
        for n, h in enumerate(hermites):
            if n > 0:
                f *= 2 * n
            scale_factor = tf.exp(-tf.square(x) / exp_denom) / np.sqrt(f)
            hermites[n] = scale_factor * h
        return hermites

    def get_domain(self) -> Tuple[float, float]:
        return (-np.inf, np.inf)

    def get_normalization_factor(self, order: int) -> int:
        return 1

    def get_weighting_fn(self, x: tf.Tensor) -> tf.Tensor:
        return tf.ones_like(x)

    def __repr__(self):
        return 'GaussHermitePolyBuilder(%s)' % str(self.stddev).rstrip('0')


@gin.configurable(module='kblocks.ops')
class GegenbauerPolynomialBuilder(RecursiveOrthogonalPolynomialBuilder):

    def __init__(self, lam: Union[Num, tf.Tensor] = 0.75):
        self.lam = lam

    def get_p1(self, x: tf.Tensor) -> tf.Tensor:
        return 2 * x if self.lam == 0 else 2 * self.lam * x

    def get_next(self, x: tf.Tensor, pn2: tf.Tensor, pn1: tf.Tensor,
                 n: int) -> tf.Tensor:
        if n == 2 and self.lam == 0:
            return x * pn1 - 1
        else:
            rhs = 2 * (n - 1 + self.lam) * x * pn1 - (n - 2 +
                                                      2 * self.lam) * pn2
            return rhs / n

    def get_domain(self) -> Tuple[int, int]:
        return (-1, 1)

    def get_weighting_fn(self, x: tf.Tensor) -> tf.Tensor:
        return (1 - x**2)**(self.lam - 0.5)

    def get_normalization_factor(self, order: int) -> float:
        if self.lam == 0:
            if order == 0:
                return np.pi
            else:
                return 2 * np.pi / order**2
        else:
            from scipy.special import gamma
            numer = np.pi * 2**(1 - 2 * self.lam) * gamma(order + 2 * self.lam)
            denom = (order + self.lam) * factorial(order) *\
                gamma(self.lam)**2
            return numer / denom

    def __repr__(self):
        return 'GegenbauerPolyBuilder(%s)' % str(self.lam).rstrip('0')


def _total_order_num_out(num_dims: int, max_order: int) -> int:
    from scipy.special import comb
    return int(comb(num_dims + max_order, max_order))
    # if num_dims == 0 or max_order == 0:
    #     return 1
    # else:
    #     return sum(
    #         _total_order_num_out(num_dims - 1, max_order - i)
    #         for i in range(max_order + 1))


@gin.configurable(module='kblocks.ops')
class NdPolynomialBuilder(object):

    def __init__(self,
                 max_order: int = 3,
                 is_total_order: bool = True,
                 base_builder: Optional[PolynomialBuilder] = None):
        if base_builder is None:
            base_builder = GeometricPolynomialBuilder()
        assert (callable(base_builder))
        self._base_builder: PolynomialBuilder = base_builder
        self._max_order = max_order
        self._is_total_order = is_total_order

    def num_out(self, num_dims: int):
        if self._is_total_order:
            return _total_order_num_out(num_dims, self._max_order) - 1
        else:
            return num_dims * self._max_order - 1

    def output_shape(self, input_shape: Sequence[int],
                     axis=-1) -> Tuple[int, ...]:
        s = list(input_shape)
        s[axis] = self.num_out(s[axis])
        return tuple(s)

    def __call__(self, coords: tf.Tensor, axis: int = -1) -> tf.Tensor:
        if (self._max_order == 1 and self._is_total_order and
                isinstance(self._base_builder, GeometricPolynomialBuilder)):
            return coords
        single_polys = []
        coords = tf.unstack(coords, axis=axis)
        for x in coords:
            polys = self._base_builder(x, self._max_order + 1)
            single_polys.append(enumerate(polys))

        outputs = []
        for ordered_polys in itertools.product(*single_polys):
            orders, polys = zip(*ordered_polys)
            total_order = sum(orders)

            if total_order == 0 or (self._is_total_order and
                                    total_order > self._max_order):
                continue
            outputs.append(tf.reduce_prod(tf.stack(polys, axis=-1), axis=-1))
        outputs = tf.stack(outputs, axis=axis)
        assert (outputs.shape[axis] == self.num_out(len(coords)))
        return outputs


_builder_factories = {
    'geo': GeometricPolynomialBuilder,
    'cheb': ChebyshevPolynomialBuilder.from_kind,
    'che1': FirstChebyshevPolynomialBuilder,
    'che2': SecondChebyshevPolynomialBuilder,
    'gh': GaussianHermitePolynomialBuilder,
    'her': HermitePolynomialBuilder,
    'geg': GegenbauerPolynomialBuilder,
    'leg': LegendrePolynomialBuilder,
}


def _builder_from_dict(name: str, **kwargs):
    return _builder_factories[name](**kwargs)


def deserialize_builder(obj):
    if obj is None:
        return None
    elif isinstance(obj, PolynomialBuilder):
        return obj
    elif isinstance(obj, six.string_types):
        return _builder_from_dict(obj)
    else:
        assert (isinstance(obj, dict))
        return _builder_from_dict(**obj)


@gin.configurable(module='kblocks.ops')
def get_nd_polynomials(
        coords: tf.Tensor,
        max_order: int = 3,
        is_total_order: bool = True,
        base_builder: Optional[Union[PolynomialBuilder, str]] = None,
        axis: int = -1) -> tf.Tensor:
    builder = deserialize_builder(base_builder)
    return NdPolynomialBuilder(max_order, is_total_order, builder)(coords,
                                                                   axis=axis)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    keys = sorted(_builder_factories)
    keys = [k for k in keys if k not in ('her', 'cheb')]
    builders = [_builder_factories[k]() for k in keys]
    x = tf.linspace(-1.0, 1.0, 101)
    xnp = x.numpy()
    min_order = 1
    max_order = 4
    polys = [b(x, max_order)[min_order:] for b in builders]
    # polys = [[pi * (1 - tf.abs(x)) for pi in p] for p in polys]

    for i in range(max_order - min_order):
        plt.figure()
        lines = [
            plt.plot(xnp, p[i].numpy(), label=l) for p, l in zip(polys, keys)
        ]
        plt.legend()
        plt.title('order = {}'.format(i + min_order))

    plt.show()
