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
import os
import time
import random
import signal
from threading import Lock, Event

LOCK = Lock()
MODEL_ID = -1

MODEL_NAMES = [l.strip().lower().replace(' ', '_') for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_names.txt')).readlines()]
random.shuffle(MODEL_NAMES)

def next_user_friendly_model_id():
    global LOCK, MODEL_ID
    with LOCK:
        MODEL_ID += 1
        model_name = MODEL_NAMES[MODEL_ID]
        if MODEL_ID >= len(MODEL_NAMES):
            model_name = model_name + "_" + int(MODEL_ID/len(MODEL_NAMES))
        return model_name

def reset_user_friendly_model_id():
    global LOCK, MODEL_ID
    with LOCK:
        MODEL_ID = -1


exit_event = Event()    
for sig in ('TERM', 'HUP', 'INT'):
    signal.signal(getattr(signal, 'SIG'+sig), lambda: exit_event.set())


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
