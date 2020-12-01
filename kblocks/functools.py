import functools

import gin

partial = gin.external_configurable(functools.partial, module="functools")
