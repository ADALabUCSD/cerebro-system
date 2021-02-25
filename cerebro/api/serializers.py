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


############### Hyperparameters #################
param_def = api.model('ParamDef', {
    # 'id': fields.String(readonly=True, description='Hyperparameter definition UID'),
    'name': fields.String(required=True, description='Hyperparameter name'),
    'param_type': fields.String(required=True, description='Hyperparameter type', enum=['categorical', 'int', 'float', 'log']),
    'values': fields.List(fields.String, required=False, description='Categorical values in the case of a categorical hparam type'),
    'min_val': fields.Float(required=False, description='Minimum value used for int, float, and log hparam types'),
    'max_val': fields.Float(required=False, description='Maximum value used for int, float, and log hparam types'),
    'count': fields.Integer(required=False, description='Number of values to be taken from the range [min, max]. For int and float values are evenly chosen. For log values are chosen logarithmically evenly'),
    'base': fields.Integer(required=False, description='Base value for log hparam type'),
})

param_val = api.model('ParamVal', {
    # 'id': fields.String(readonly=True, description='Hyperparameter value UID'),
    'name': fields.String(required=True, description='Hyperparameter name'),
    'value': fields.String(required=True, description='Hyperparameter value'),
    'value_type': fields.String(required=True, description='Hyperparameter value type', enum=['str', 'int', 'float'])
})

#################### Metrics ####################
metric = api.model('Metric', {
    'name': fields.String(readonly=True, description='Metric name'),
    'metric_type': fields.String(readonly=True, description='Metric type', enum=['train', 'validation']),
    'values': fields.List(fields.Float, readonly=True, description='Metric values list for every epoch')
})


#################### Model ######################
model = api.model('Model', {
    'id': fields.String(readonly=True, description='Model UID'),
    'experiment_id': fields.String(required=True, description='Experiment UID'),
    'creation_time': fields.DateTime(readonly=True, description='Experiment creation time'),
    'last_update_time': fields.DateTime(readonly=True, description='Experiment last update time'),
    'status': fields.String(readonly=True, description='Model status', enum=['created', 'running', 'failed', 'completed']),
    'param_vals': fields.List(fields.Nested(param_val), required=True, description='Hyperparameter values'),
    'metrics': fields.List(fields.Nested(metric), readonly=True, description='Model training metrics')
})


################### Experiment ##################
experiment = api.model('Experiment', {
    'id': fields.String(readonly=True, description='Experiment UID'),
    'name': fields.String(required=True, description='Experiment name'),
    'description': fields.String(description='Experiment description'),
    'creation_time': fields.DateTime(readonly=True, description='Experiment creation time'),
    'last_update_time': fields.DateTime(readonly=True, description='Experiment last update time'),
    'status': fields.String(readonly=True, description='Experiment status', enum=['created', 'running', 'failed', 'completed']),
    'param_defs': fields.List(fields.Nested(param_def), required=True, description='Hyperparameter definitions'),
    'feature_columns': fields.List(fields.String, required=True, description='Training data feature columns names'),
    'label_columns': fields.List(fields.String, required=True, description='Training data label column names'),
    'max_train_epochs': fields.Integer(required=True, description='Maximum number of training epochs for any model'),
    'training_data_prefix_path': fields.String(required=True, description='Training data prefix path'),
    'executable_entrypoint': fields.String(required=True, description='Estimator generator function in the form of <module_name>:<function_name>'),
    'models': fields.List(fields.Nested(model), readonly=True, description='Models in this experiment')
})
