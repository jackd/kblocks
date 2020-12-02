# [Injectable tf.keras Blocks](https://github.com/jackd/kblocks)

This package provides a rapid prototyping environment for running deep learning experiments. There are three main components:

- Injectable keras components - "blocks" - which are just regular `keras` and other `tf` classes/methods wrapped in `gin.configurable`, including some custom ones I feel are missing from core `keras`;
- a convenient command-line interface that allows `main` functions to be configured via `gin`; and
- common main functions accessible via this CLI.

## Installation

```bash
pip install tensorflow>=2.3  # not included in requirements.txt - could be tf-nightly
git clone https://github.com/jackd/kblocks.git
cd kblocks
pip install -r requirements.txt
pip install -e .
```

Some examples require [tensorflow-datasets](https://github.com/tensorflow/datasets)

```bash
pip install tensorflow-datasets
```

The profiling experience is best with

```bash
pip install tensorboard_plugin_profile
```

This will also require `libcupti` on your `$LD_LIBRARY_PATH`. Ensure you version consistent with your tensorflow installation can be found

```bash
/sbin/ldconfig -N -v $(sed 's/:/ /g' <<< $LD_LIBRARY_PATH) | grep libcupti
```

This repository depends on the following packages not available on github:

- [meta-model](https://github.com/jackd/meta-model)
- [tfrng](https://github.com/jackd/tfrng)
- [wtftf](https://github.com/jackd/wtftf)

## Examples

See [examples/image_cls](examples/image_cls) for examples.

```bash
cd examples/image_cls
# basic example without configuration
python fit_simple_cifar10.py
# Trainable examples configured with gin
./scripts/mnist-reg.sh
./scripts/profile-cifar10.sh
./scripts/benchmark-cifar10.sh
```

## Reproducibility

The aim is for training with [fit](kblocks/models/fit.py) to lead to reproducible results. In order to achieve this:

1. training must be performed in a single step (i.e. no restarting from earlier `fit`s see below);
2. `TfConfig.seed` must be configured;
3. data augmentation functions must use `tfrng` ops, rather than `tf.random` ops, and maps should use `tfrng.data.stateless_map`; and
4. models must contain only deterministic functions - see [NVIDIA/framework-determinism](https://github.com/NVIDIA/framework-determinism) for full details.

Work on supporting pre-emptible training (relaxing constraint 1 above) is on-going.

## Further Reading

Go check out the [gin user guide](https://github.com/google/gin-config/blob/master/docs/index.md) for more examples of how best to use this powerful framework. Happy configuring!

## Projects using `kblocks`

- Implementations from [Sparse Convolutions on Continuous Domains](https://github.com/jackd/sccd.git):
  - [Point Cloud Network](https://github.com/jackd/pcn.git)
  - [Event Convolution Network](https://github.com/jackd/ecn.git)

## TODO

- refactor [polynomials](kblocks/ops/polynomials) into separate repo?
- configure `%start_time` macro?
- make default experiment dir based on `system_time`?
- does `tfrecords_cache` produce datasets with checkpoint-able datasets?
