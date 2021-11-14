from os import stat

import keras_tuner
from .base import CerebroOracle
from ..sparktuner import SparkTuner
from keras_tuner.engine import trial as trial_lib
from ...tune.base import ModelSelection
from typing import Optional, List, Any, Dict

class RandomSearchOracle(CerebroOracle):
    def __init__(
        self, 
        objective, 
        max_trials=None, 
        hyperparameters=None, 
        allow_new_entries=True, 
        tune_new_entries=True, 
        seed=None):

        super().__init__(objective, max_trials=max_trials, hyperparameters=hyperparameters, allow_new_entries=allow_new_entries, tune_new_entries=tune_new_entries, seed=seed)

    def populate_space(self, trial_id):
        values = self._random_values()
        if values is None:
            return {"status": trial_lib.TrialStatus.STOPPED, "values": None}
        return {"status": trial_lib.TrialStatus.RUNNING, "values": values}

    def create_trials(self, n):
        trials = []
        for i in range(n):
            trial_id = trial_lib.generate_trial_id()
            if self.max_trials and len(self.trials) >= self.max_trials:
                status = trial_lib.TrialStatus.STOPPED
                values = None
            else:
                response = self.populate_space(trial_id)
                status = response["status"]
                values = response["values"] if "values" in response else None
            hps = self.hyperparameters.copy()
            hps.values = values or {}
            trial = trial_lib.Trial(
                hyperparameters=hps, trial_id=trial_id,
                status=status
            )

            if status == trial_lib.TrialStatus.RUNNING:
                self.trials[trial_id] = trial
                self.start_order.append(trial_id)
                self._save_trial(trial)
                self.save()
            trials.append(trial)
        return trials
    
    def _init_search_space(self):
        pass

class RandomSearch(SparkTuner):
    def __init__(
        self, 
        oracle, 
        hypermodel, 
        model_selection: ModelSelection,
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
        oracle = RandomSearchOracle(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
        )
        super().__init__(oracle, hypermodel, model_selection, **kwargs)
        
