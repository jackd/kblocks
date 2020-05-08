"""
Example usage:

```python
import tensorflow as tf
from kblocks.keras import custom_objects

@custom_objects.register
def swish(x):
    return x / (1 + tf.exp(-x))


with custom_objects.scope():
    activation = tf.keras.activations.get('swish')
print(activation)
```

Note there is also the global `tf.keras.utils.get_custom_objects`.
"""
from typing import Union, Callable
import tensorflow as tf
from collections import MutableMapping


class CustomObjectRegister(MutableMapping):
    def __init__(self):
        self._objects = {}

    def __getitem__(self, key):
        return self._objects[key]

    def __iter__(self):
        return iter(self._objects)

    def __len__(self):
        return len(self._objects)

    def __contains__(self, key):
        return key in self._objects

    def __delitem__(self, key):
        raise NotImplementedError("Cannot delete items from CustomObjectRegister")

    def _validate_not_present(self, key):
        if key in self._objects:
            raise KeyError(f"Value already exists at key {key}")

    def __setitem__(self, key, value):
        self._validate_not_present(key)
        self._objects[key] = value

    def scope(self):
        return tf.keras.utils.custom_object_scope(self._objects)

    def register(self, fn_or_name: Union[Callable, str]):
        if callable(fn_or_name):
            if not hasattr(fn_or_name, "__name__"):
                raise ValueError("Cannot register callable without a __name__ attr")
            name = fn_or_name.__name__
            fn = fn_or_name
            self[name] = fn
            return fn
        if isinstance(fn_or_name, str):
            name = fn_or_name
            self._validate_not_present(name)

            def f(fn):
                self[name] = fn
                return fn

            return f
        else:
            raise ValueError(f"fn_or_name must be a callable or str, got {fn_or_name}")


_objects = CustomObjectRegister()

register = _objects.register
scope = _objects.scope
