import keras_tuner
import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import keras_tuner
import numpy as np

class CerebroOracle(keras_tuner.Oracle):
    """
    Base Oracle that supports multiple trial configuration

    Args:
        objective: A string or `keras_tuner.Objective` instance. If a string,
            the direction of the optimization (min or max) will be inferred.
        max_trials: Integer, the total number of trials (model configurations)
            to test at most. Note that the oracle may interrupt the search
            before `max_trial` models have been tested if the search space has
            been exhausted.
        hyperparameters: Optional `HyperParameters` instance. Can be used to
            override (or register in advance) hyperparameters in the search
            space.
        tune_new_entries: Boolean, whether hyperparameter entries that are
            requested by the hypermodel but that were not specified in
            `hyperparameters` should be added to the search space, or not. If
            not, then the default value for these parameters will be used.
            Defaults to True.
        allow_new_entries: Boolean, whether the hypermodel is allowed to
            request hyperparameter entries not listed in `hyperparameters`.
            Defaults to True.
        seed: Int. Random seed.
    """

    def __init__(
        self, 
        objective, 
        max_trials=None, 
        hyperparameters=None, 
        allow_new_entries=True, 
        tune_new_entries=True, 
        seed=None
        ):
        super().__init__(
            objective, 
            max_trials=max_trials,
            hyperparameters=hyperparameters, allow_new_entries=allow_new_entries, tune_new_entries=tune_new_entries, 
            seed=seed
            )

    def _init_search_space(self):
        """
        Init the entire search space for the hyperparameters
        This function should be called in the Tuner.search since parameters for the architecture are now known until seeing the data
        """
        raise NotImplementedError
    
    def create_trials(self, n):
        raise NotImplementedError