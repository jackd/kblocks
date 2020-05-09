import os
from typing import Iterable, Mapping, Optional, Union

import gin

from kblocks.gin_utils.config import fix_bindings, fix_paths
from kblocks.gin_utils.path import enable_relative_includes, enable_variable_expansion

_GIN_SUMMARY = """
# --cwd={cwd}
# --incl_rel={incl_rel}
# --expand_vars={expand_vars}

# -------------------
# CONFIG FILES
{config_files}

# -------------------
# BINDINGS
{bindings}
"""


class GinSummary:
    """
    Class for summarizing gin configuration.

    Args:
        cwd: current working directory.
        incl_rel: True indicates `enable_relative_includes` should be called.
        expand_vars: True indicates `enable_variable_expansion` should be
            called.
        config_files: path or paths to config files to be included.
        bindings: additional bindings to be set after files are parsed.
    """

    def __init__(
        self,
        cwd: Optional[str] = None,
        incl_rel: bool = True,
        expand_vars: bool = True,
        config_files: Union[str, Iterable[str]] = (),
        bindings: Union[str, Iterable[str]] = (),
    ):
        self.cwd = os.getcwd() if cwd is None else cwd
        self.incl_rel = incl_rel
        self.expand_vars = expand_vars
        self.config_files = fix_paths(config_files)
        self.bindings = fix_bindings(bindings)

    def get_config(self):
        """Return a dictionary representation of this object."""
        return dict(
            cwd=self.cwd,
            incl_rel=self.incl_rel,
            expand_vars=self.expand_vars,
            config_files=self.config_files,
            bindings=self.bindings,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Union[bool, str]]):
        return cls(**config)

    def pretty_format(self):
        """Multi-line human readable string representation."""
        config = self.get_config()
        files = config["config_files"]
        assert isinstance(files, list)
        config["config_files"] = "\n".join(files)
        return _GIN_SUMMARY.format(**config)

    def enable_path_options(self):
        """Enable relative/expansion options depending on constructor values."""
        if self.incl_rel:
            enable_relative_includes()
        if self.expand_vars:
            enable_variable_expansion()

    def parse(self, finalize: bool = True):
        """Parse files/bindings provided in constructor."""
        gin.parse_config_files_and_bindings(
            self.config_files, self.bindings, finalize_config=finalize
        )
