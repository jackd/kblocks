python -m kblocks '$KB_CONFIG/benchmark/trainable.gin' \
    configs/base.gin \
    configs/models/simple.gin \
    configs/data/cifar10.gin \
    --bindings='use_rngs=False'  # Generator.split doesn't work in graph mode :(
