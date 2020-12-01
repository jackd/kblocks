import gin

from kblocks.gin_utils.path import enable_variable_expansion


@gin.configurable
def print_kwargs(**kwargs):
    print("--------")
    for k in sorted(kwargs):
        print(f"{k}: {kwargs[k]}")
    print("--------")


bindings = """
import kblocks.path
import kblocks.configs

include "$KB_CONFIG/utils/path.gin"
print_kwargs.root_dir = %root_dir
print_kwargs.problem_dir = %problem_dir
print_kwargs.family_dir = %family_dir
print_kwargs.variant_dir = %variant_dir
print_kwargs.experiment_dir = %experiment_dir

root_dir = @kb.path.expand()
root_dir/expand.path = '~/custom_dir'
run = 3
"""

enable_variable_expansion()
gin.parse_config(bindings)
print_kwargs()
