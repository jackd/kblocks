"""
Basic command line interface utilities for working with gin and abs.logging.

Importing this adds 4 command line args.

- config_files: gin files to include
- bindings: additional configurations applied after files are parsed
- incl_rel (default True): enable relative paths in includes if True
- expand_vars (default True): enable environment variables in paths if True

`get_gin_summary` gets a GinSummary object from the command line (interpretting)
unused positional arguments as additional config files).
"""


import os
from typing import Callable, Optional

import gin
from absl import flags, logging

from kblocks import tf_config, utils

flags.DEFINE_multi_string(
    "config_files", [], "config files appended to positional args."
)
flags.DEFINE_multi_string(
    "bindings", [], "Newline separated list of gin parameter bindings."
)
flags.DEFINE_boolean(
    "incl_rel", default=True, help="Whether or not to enable_relative_includes"
)
flags.DEFINE_boolean(
    "expand_vars", default=True, help="Whether or not to enable vars/user in includes"
)


@gin.configurable(module="kb")
def log_dir(root_dir):
    return os.path.join(root_dir, "logs")


def _unique_prog_name(log_dir, base_prog_name):
    prog_name = base_prog_name
    info_path = os.path.join(log_dir, "{}.INFO".format(prog_name))
    i = 0
    while os.path.isfile(info_path):
        i += 1
        prog_name = "{}{}".format(base_prog_name, i)
        info_path = os.path.join(log_dir, "{}.INFO".format(prog_name))
    return prog_name


@gin.configurable(module="kb")
def logging_config(
    to_file: bool = True,
    log_dir: Optional[str] = None,
    program_name: str = "kblocks",
    force_unique: bool = True,
):
    if to_file and log_dir is not None:
        log_dir = os.path.expanduser(os.path.expandvars(log_dir))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.info("Logging to {}".format(log_dir))
        if force_unique:
            program_name = _unique_prog_name(log_dir, program_name)
        logging.get_absl_handler().use_absl_log_file(
            log_dir=log_dir, program_name=program_name
        )


def get_gin_summary(argv):
    """
    Collect GinSummary from command line args

    Args:
        argv: clargs that weren't passed by absl. Interpretted as config files
        finalize_config: if True, config is finalized after parsing.

    Returns:
        GinSummary object
    """
    from kblocks.gin_utils.summary import GinSummary

    FLAGS = flags.FLAGS
    return GinSummary(
        os.getcwd(),
        FLAGS.incl_rel,
        FLAGS.expand_vars,
        argv[1:] + FLAGS.config_files,
        FLAGS.bindings,
    )


@gin.configurable(module="kb")
def main(fn: Optional[Callable] = None):
    if fn is None:
        logging.error("`main.fn` is not configured.")
    if not callable(fn):
        raise ValueError("`main.fn` is not callable.")
    return fn()


def summary_main(gin_summary):
    gin_summary.enable_path_options()
    gin_summary.parse(finalize=True)
    logging_config()
    logging.info(gin_summary.pretty_format())
    utils.proc()
    tf_config.TfConfig().configure()
    return main()
