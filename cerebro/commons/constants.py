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

## Experiment and Model statusses
CREATED_STATUS = 'created'
RUNNING_STATUS = 'running'
FAILED_STATUS = 'failed'
STOPPED_STATUS = 'stopped'
COMPLETED_STATUS = 'completed'

## Hyperparameter types
HP_CHOICE = 'hp_choice'
HP_LOGUNIFORM = 'hp_loguniform'
HP_QLOGUNIFORM = 'hp_qloguniform'
HP_QUNIFORM = 'hp_quniform'
HP_UNIFORM = 'hp_uniform'

## Hyperparameter dtypes
DTYPE_STR = 'dtype_str'
DTYPE_INT = 'dtype_int'
DTYPE_FLOAT = 'dtype_float'

## Hyperparameter search procedures
MS_GRID_SEARCH = 'GridSearch'
MS_RANDOM_SEARCH = 'RandomSearch'
MS_HYPEROPT_SEARCH = 'HyperOpt'
