# Copyright 2021 Supun Nakandala, and Arun Kumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from flask_restplus import fields
from .restplus import api
from ..commons.constants import *

state_enums = [CREATED_STATUS, RUNNING_STATUS, FAILED_STATUS, STOPPED_STATUS, CREATED_STATUS]
param_type_enums = [HP_CHOICE, HP_LOGUNIFORM, HP_QLOGUNIFORM, HP_QUNIFORM, HP_UNIFORM]
param_dtype_enums = [DTYPE_STR, DTYPE_INT, DTYPE_FLOAT]
model_selection_algo_enums = [MS_GRID_SEARCH, MS_RANDOM_SEARCH, MS_HYPEROPT_SEARCH]


############### Hyperparameters #################
param_def = api.model('ParamDef', {
    'name': fields.String(required=True, description='Hyperparameter name'),
    'param_type': fields.String(required=True, description='Hyperparameter type', enum=param_type_enums),
    'choices': fields.String(required=False, description='Comma separated list of values in the case of a {} hparam type'.format(HP_CHOICE)),
    'min': fields.Float(required=False, description='Minimum value'),
    'max': fields.Float(required=False, description='Maximum value'),
    'q': fields.Float(required=False, description='Quantum'),
    'dtype': fields.String(required=True, description='Parameter dtype', enum=param_dtype_enums, default=DTYPE_STR)
})

param_val = api.model('ParamVal', {
    'name': fields.String(required=True, description='Hyperparameter name'),
    'value': fields.String(required=True, description='Hyperparameter value'),
    'dtype': fields.String(readonly=True, description='Hyperparameter value dtype')
})

#################### Metrics ####################
metric = api.model('Metric', {
    'name': fields.String(readonly=True, description='Metric name'),
    'values': fields.String(readonly=True, description='Comma separated list of metric values list for every epoch')
})


#################### Model ######################
model = api.model('Model', {
    'id': fields.String(readonly=True, description='Model UID'),
    'name': fields.String(readonly=True, description='Userfriendly model name'),
    'exp_id': fields.String(required=True, description='Experiment UID'),
    'creation_time': fields.DateTime(readonly=True, description='Experiment creation time'),
    'last_update_time': fields.DateTime(readonly=True, description='Experiment last update time'),
    'status': fields.String(readonly=True, description='Model status', enum=state_enums),
    'num_trained_epochs': fields.Integer(readonly=True, description='Current number of trained epochs for the model'),
    'max_train_epochs': fields.Integer(required=True, description='Maximum number of training epochs for the model'),
    'warm_start_model_id': fields.String(required=False, description='UID of the warm starting model in the case of a model clone'),
    'param_vals': fields.List(fields.Nested(param_val), required=True, description='Hyperparameter values'),
    'metrics': fields.List(fields.Nested(metric), readonly=True, description='Model training metrics'),
    'exception_message': fields.String(readonly=True, description='Exception message in the case of an model initialization failure')
})


################### Experiment ##################
experiment = api.model('Experiment', {
    'id': fields.String(readonly=True, description='Experiment UID'),
    'name': fields.String(required=True, description='Experiment name'),
    'description': fields.String(description='Experiment description'),
    'clone_model_id': fields.String(description='Cloned model UID'),
    'warm_start_from_cloned_model': fields.Boolean(description='Whether to warmstart the weights from the cloned model'),
    'creation_time': fields.DateTime(readonly=True, description='Experiment creation time'),
    'last_update_time': fields.DateTime(readonly=True, description='Experiment last update time'),
    'status': fields.String(readonly=True, description='Experiment status', enum=state_enums),
    'model_selection_algorithm': fields.String(required=True, description='Model selection algorithm', enum=model_selection_algo_enums),
    'max_num_models': fields.Integer(required=False, description='Maximum number of models to be explored by the model selection algorithm. Needed for RandomSearch, HyperOpt'),
    'param_defs': fields.List(fields.Nested(param_def), required=True, description='Hyperparameter definitions'),
    'feature_columns': fields.String(required=True, description='Comma separated list of training data feature columns names'),
    'label_columns': fields.String(required=True, description='Comma separated list of training data label column names'),
    'max_train_epochs': fields.Integer(required=True, description='Maximum number of training epochs for any model'),
    'data_store_prefix_path': fields.String(required=True, description='Data store prefix path'),
    'executable_entrypoint': fields.String(required=True, description='Estimator generator function in the form of <module_name>:<function_name>'),
    'models': fields.List(fields.Nested(model), readonly=True, description='Models in this experiment'),
    'exception_message': fields.String(readonly=True, description='Exception message in the case of an experiment failure')
})
