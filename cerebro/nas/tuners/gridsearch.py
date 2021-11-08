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

import copy
from typing import Optional, Dict, List, Any
import keras_tuner
from keras_tuner.engine import oracle
from numpy.lib.function_base import copy
from ..sparktuner import SparkTuner
from ..hphpmodel import HyperHyperModel

class GridSearchOracle(keras_tuner.Oracle):
    def __init__(
        self, 
        objective, 
        initial_hps=None,
        max_trials=None, 
        hyperparameters=None, 
        allow_new_entries=True, 
        tune_new_entries=True, 
        seed=None):

        self.initial_hps = copy.deepcopy(initial_hps) or []
        self.visited_hps = [False] * len(self.initial_hps)

        super().__init__(objective, max_trials=max_trials, hyperparameters=hyperparameters, allow_new_entries=allow_new_entries, tune_new_entries=tune_new_entries, seed=seed)

        self.grid_space = self._init_space()

    # def _init_space(self):


class GridSearch(SparkTuner):
    """
    GridSearch with multiple trials training
    """

    def __init__(
        self,
        hypermodel: HyperHyperModel,
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
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
