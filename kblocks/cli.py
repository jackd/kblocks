from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl import logging
import gin
from typing import Optional, Callable

flags.DEFINE_multi_string('config_files', [],
                          'config files appended to positional args.')
flags.DEFINE_multi_string('bindings', [],
                          'Newline separated list of gin parameter bindings.')
flags.DEFINE_boolean('incl_rel',
                     default=True,
                     help='Whether or not to enable_relative_includes')
flags.DEFINE_boolean('expand_vars',
                     default=True,
                     help='Whether or not to enable vars/user in includes')


@gin.configurable(module='kb')
def log_dir(root_dir):
    return os.path.join(root_dir, 'logs')


def _unique_prog_name(log_dir, base_prog_name):
    prog_name = base_prog_name
    info_path = os.path.join(log_dir, '{}.INFO'.format(prog_name))
    i = 0
    while os.path.isfile(info_path):
        i += 1
        prog_name = '{}{}'.format(base_prog_name, i)
        info_path = os.path.join(log_dir, '{}.INFO'.format(prog_name))
    return prog_name


@gin.configurable(module='kb')
def logging_config(to_file: bool = True,
                   log_dir: Optional[str] = None,
                   program_name: str = 'kblocks',
                   force_unique: bool = True):
    if to_file and log_dir is not None:
        log_dir = os.path.expanduser(os.path.expandvars(log_dir))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.info('Logging to {}'.format(log_dir))
        if force_unique:
            program_name = _unique_prog_name(log_dir, program_name)
        logging.get_absl_handler().use_absl_log_file(log_dir=log_dir,
                                                     program_name=program_name)


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
    return GinSummary(os.getcwd(), FLAGS.incl_rel, FLAGS.expand_vars,
                      argv[1:] + FLAGS.config_files, FLAGS.bindings)


@gin.configurable(module='kb')
def main(fn: Optional[Callable] = None):
    if fn is None:
        logging.error('`main.fn` is not configured.')
    if not callable(fn):
        raise ValueError('`main.fn` is not callable.')
    fn()
