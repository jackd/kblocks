# demonstrates how to use bindings
python -m kblocks '$KB_CONFIG/trainable/fit.gin' \
    configs/base.gin \
    configs/models/simple.gin \
    configs/data/mnist.gin \
    configs/variants/reg.gin
