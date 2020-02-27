import gin
import kerastuner as kt

Tuner = gin.external_configurable(kt.Tuner, module='kt')
RandomSearch = gin.external_configurable(kt.RandomSearch, module='kt')
BayesianOptimization = gin.external_configurable(kt.BayesianOptimization,
                                                 module='kt')
Hyperband = gin.external_configurable(kt.Hyperband, module='kt')

Objective = gin.external_configurable(kt.Objective, module='kt')
