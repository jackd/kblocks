from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gin


@gin.configurable(module='kb.path')
def model_dir(root_dir='~/kblocks',
              problem_id='default_prob',
              model_id='default_model',
              variant_id='v0',
              run=0):
    return os.path.join(root_dir, problem_id, model_id, variant_id,
                        'run-{:02d}'.format(run))
