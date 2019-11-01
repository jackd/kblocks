from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import logging
import contextlib
from typing import Union, Iterable, List

KB_CONFIG_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', 'configs'))
if 'KB_CONFIG' in os.environ:
    logging.warning('KB_CONFIG environment variable defined. '
                    '`kblocks.gin_utils.parse_kb_config` may act surprisingly')
else:
    os.environ['KB_CONFIG'] = KB_CONFIG_DIR


@contextlib.contextmanager
def change_dir_context(path: str):
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


def fix_paths(config_files: Union[str, Iterable[str]]) -> List[str]:
    """
    Convert a string/list of strings into a list of strings in canonical form.

    In order:
        - converts a single string to a single-element list
        - concatenates the result of splitting over new lines and comma
        - filters out empty strings
    """
    import numpy as np
    if config_files == 0:
        return list(config_files)
    if isinstance(config_files, str):
        config_files = [config_files]
    config_files = np.concatenate([c.split('\n') for c in config_files])
    config_files = np.concatenate([c.split(',') for c in config_files])
    config_files = (c.strip() for c in config_files)
    config_files = (c for c in config_files if c != '')
    config_files = (  # add missing .gin extension
        p if p.endswith('.gin') else '{}.gin'.format(p) for p in config_files)
    return [p for p in config_files if p.strip() != '']


def fix_bindings(bindings: Union[str, Iterable[str]]) -> str:
    """Convert a string/list of strings into a single string of bindings."""
    if isinstance(bindings, str):
        return bindings
    else:
        return '\n'.join(bindings)
