# Copyright 2021 Zijian He
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

import collections
from typing import Optional, Dict, List, Any
import numpy as np
import itertools

import keras_tuner
from keras_tuner.engine import oracle
from tensorflow.python.distribute import values
from tensorflow.python.ops.gen_control_flow_ops import switch

from ..sparktuner import SparkTuner
from .base import CerebroOracle
from keras_tuner.engine import hyperparameters

class GridSearchOracle(CerebroOracle):
    def __init__(
        self, 
        objective, 
        initial_hps=None,
        max_trials=None, 
        hyperparameters=None, 
        allow_new_entries=True, 
        tune_new_entries=True, 
        seed=None):

        self.initial_hps = initial_hps or []
        self.visited_hps = [False] * len(self.initial_hps)

        super().__init__(objective, max_trials=max_trials, hyperparameters=hyperparameters, allow_new_entries=allow_new_entries, tune_new_entries=tune_new_entries, seed=seed)


    def _validate_hyperparameters(self):
        # TODO Any hp checkings
        # for hp in self.hyperparameters.space:
        #     if not type(hp) is   not hp.sampling is None:
        #         raise Exception('Gridsearch does not support sampling algorithms for the hyperparameters at the current stage')
        pass

    def _grid_generation(self, hps):
        grid_space = collections.defaultdict(list)
        for hp in hps.space:
            if type(hp) is hyperparameters.Choice:
                grid_space[hp.name].extend(hp.values)
            if type(hp) is hyperparameters.Int:
                grid_space[hp.name].extend([i for i in range(hp.min_vlaue, hp.max_value, hp.step)])
            if type(hp) is hyperparameters.Float:
                if hp.step is None:
                    raise Exception('Need to specify step for the floating hp to initialize grid search')
                else:
                    grid_space[hp.name].extend([i for i in np.arange(hp.min_vlaue, hp.max_value, hp.step)])
            if type(hp) is hyperparameters.Boolean:
                grid_space[hp.name].extend([True, False])
            if type(hp) is hyperparameters.Fixed:
                grid_space[hp.name].append(hp.default)
        
        return [dict(zip(grid_space.keys(), values)) for values in itertools.product(*grid_space.values())]


                

    def init_search_space(self):
        """
        return hyperparameter combinations E.g. Given a hp as
        {
            "learning_rate": [0.1, 0.01],
            "batch_size": [1, 2]
        }
        the entire search space should be:
        [
            {"learning_rate": 0.1, "batch_size":1}
            {"learning_rate": 0.1, "batch_size":2}
            {"learning_rate": 0.01, "batch_size":1}
            {"learning_rate": 0.01, "batch_size":2}
            ...
        ]
        """
        self._validate_hyperparameters()
        self.search_space = self._grid_generation(hps=self.hyperparameters)
        #log 

        # return super()._init_search_space()

    def create_trials(self, n):
        pass



class GridSearch(SparkTuner):
    """
    GridSearch with multiple trials training
    """

    def __init__(
        self,
        hypermodel: keras_tuner.HyperModel,
        objective: str = "val_loss",
        max_trials: int = 10,
        initial_hps: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        hyperparameters: Optional[keras_tuner.HyperParameters] = None,
        tune_new_entries: bool = True,
        allow_new_entries: bool = True,
        **kwargs
    ):
        self.seed = seed
        oracle = GridSearchOracle(
            objective=objective,
            max_trials=max_trials,
            initial_hps=initial_hps,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
        )
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
