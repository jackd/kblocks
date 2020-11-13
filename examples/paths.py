import os

import gin

from kblocks.gin_utils.config import KB_CONFIG_DIR


@gin.configurable
def print_kwargs(**kwargs):
    print("--------")
    for k in sorted(kwargs):
        print(f"{k}: {kwargs[k]}")
    print("--------")


bindings = """
print_kwargs.root_dir = %root_dir
print_kwargs.problem_dir = %problem_dir
print_kwargs.family_dir = %family_dir
print_kwargs.variant_dir = %variant_dir
print_kwargs.model_dir = %model_dir

root_dir = '~/custom_dir'
run = 3
"""

path = os.path.join(KB_CONFIG_DIR, "utils", "path.gin")
gin.parse_config_files_and_bindings([path], [bindings])
print_kwargs()
