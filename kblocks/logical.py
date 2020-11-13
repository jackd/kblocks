import gin


@gin.configurable(module="kb")
def logical_not(cond):
    return not cond


@gin.configurable(module="kb")
def logical_and(a, b):
    return a and b


@gin.configurable(module="kb")
def logical_or(a, b):
    return a or b
