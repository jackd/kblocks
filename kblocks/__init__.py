"""`tf.keras` blocks with dependency injection via `gin-config`."""
import tensorflow as tf

if not tf.version.VERSION.startswith("2"):
    raise NotImplementedError("Only tensorflow 2 supported")
