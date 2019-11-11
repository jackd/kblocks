from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gin
from kblocks.gin_utils.path import enable_variable_expansion
from kblocks.gin_utils.path import enable_relative_includes
from kblocks.gin_utils import config as _config

from typing import Union, Iterable, Mapping

_GIN_SUMMARY = '''
# --cwd={cwd}
# --incl_rel={incl_rel}
# --expand_vars={expand_vars}

# -------------------
# CONFIG FILES
{config_files}

# -------------------
# BINDINGS
{bindings}
'''


class GinSummary(object):

    def __init__(self, cwd: str, incl_rel: bool, expand_vars: bool,
                 config_files: Union[str, Iterable[str]],
                 bindings: Union[str, Iterable[str]]):
        self.cwd = cwd
        self.incl_rel = incl_rel
        self.expand_vars = expand_vars
        self.config_files = _config.fix_paths(config_files)
        self.bindings = _config.fix_bindings(bindings)

    def get_config(self):
        return dict(
            cwd=self.cwd,
            incl_rel=self.incl_rel,
            expand_vars=self.expand_vars,
            config_files=self.config_files,
            bindings=self.bindings,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, str]):
        return cls(**config)

    def pretty_format(self):
        config = self.get_config()
        files = config['config_files']
        assert (isinstance(files, list))
        config['config_files'] = '\n'.join(files)
        return _GIN_SUMMARY.format(**config)

    def enable_path_options(self):
        if self.incl_rel:
            enable_relative_includes()
        if self.expand_vars:
            enable_variable_expansion()

    def parse(self, finalize: bool = True):
        gin.parse_config_files_and_bindings(self.config_files,
                                            self.bindings,
                                            finalize_config=finalize)
