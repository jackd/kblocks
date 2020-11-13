# demonstrates how to use bindings
python -m kblocks '$KB_CONFIG/trainable/fit.gin' \
    configs/base.gin \
    configs/models/simple.gin \
    configs/data/cifar10.gin \
    --bindings='
noise_stddev = 0.1
variant_id = "aug1e-1"'
