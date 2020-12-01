import os
from typing import Iterable, List, Union

import numpy as np
from absl import logging


def try_register_config_dir(key, path):
    """
    Attempt to regiester a configuration directory or warn if failed.

    e.g. kblocks/configs/__init__.py calls
    `try_register_config_dir("KB_CONFIG", KB_CONFIG_DIR)`

    This means after importing this directory, '$KB_CONFIG/fit.gin' can be
    used from the command line or in `include` statements in gin file (assuming
    `enable_variable_expansion` has been called at some point).

    Args:
        key: environment variable key.
        path: path to configuration dir.
    """
    if key in os.environ:
        if os.environ[key] != path:
            logging.warning(
                f"{key} environment variable already defined. "
                "Config parsing may act surprisingly"
            )
    else:
        os.environ[key] = path


def fix_paths(config_files: Union[str, Iterable[str]]) -> List[str]:
    """
    Convert a string/list of strings into a list of strings in canonical form.

    In order:
        - converts a single string to a single-element list
        - concatenates the result of splitting over new lines and comma
        - filters out empty strings
    """

    if config_files == 0:
        return list(config_files)
    if isinstance(config_files, str):
        config_files = [config_files]
    config_files = np.concatenate([c.split("\n") for c in config_files])
    config_files = np.concatenate([c.split(",") for c in config_files])
    config_files = (c.strip() for c in config_files)
    config_files = (c for c in config_files if c != "")
    config_files = (  # add missing .gin extension
        p if p.endswith(".gin") else "{}.gin".format(p) for p in config_files
    )
    return [p for p in config_files if p.strip() != ""]


def fix_bindings(bindings: Union[str, Iterable[str]]) -> str:
    """Convert a string/list of strings into a single string of bindings."""
    if isinstance(bindings, str):
        return bindings
    return "\n".join(bindings)
