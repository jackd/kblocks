import matplotlib.pyplot as plt
import tensorflow as tf

from kblocks.ops import polynomials as poly

keys = [k for k in poly.builder_keys if k not in ("her", "cheb")]
builders = [poly.deserialize_builder(k) for k in keys]
x = tf.linspace(-1.0, 1.0, 101)
xnp = x.numpy()
min_order = 1
max_order = 4
polys = [b(x, max_order)[min_order:] for b in builders]

for i in range(max_order - min_order):
    plt.figure()
    lines = [plt.plot(xnp, p[i].numpy(), label=l) for p, l in zip(polys, keys)]
    plt.legend()
    plt.title("order = {}".format(i + min_order))

plt.show()
