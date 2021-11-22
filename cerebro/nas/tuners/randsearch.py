from os import stat
from h5py._hl import dataset

import keras_tuner
from tensorflow._api.v2 import data
from .base import CerebroOracle
from ..sparktuner import SparkTuner
from keras_tuner.engine import trial as trial_lib, tuner
from ...tune.base import ModelSelection, update_model_results, ModelSelectionResult
from typing import Optional, List, Any, Dict
import numpy as np

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

    def create_trials(self, n, tuner_id):
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
                self.ongoing_trials[tuner_id] = trial
                self.trials[trial_id] = trial
                self.start_order.append(trial_id)
                self._save_trial(trial)
                self.save()
            trials.append(trial)
        return trials
    
    def update_trial(self, trial_id, metrics, step):
        super().update_trial(trial_id, metrics=metrics, step=step)
    
    def _init_search_space(self):
        pass

class RandomSearch(SparkTuner):
    def __init__(
        self, 
        hypermodel, 
        parallelism,
        model_selection: ModelSelection,
        objective: str = "val_loss",
        max_trials: int = 100,
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
        super().__init__(oracle, hypermodel, parallelism, model_selection, **kwargs)

    def search(
        self, 
        epochs=None, 
        callbacks=None, 
        validation_split=0.2, 
        verbose=1,
        dataset_idx=None,
        metadata=None,
        **fit_kwargs
    ):
        """
        Search for the best HyperParameters until reach the max_trials

        # Arguments
            callbacks: A list of callback functions. Defaults to None.
            validation_split: Float.
        """
        self._display.verbose = verbose
        self.hypermodel.hypermodel.set_fit_args(validation_split, epochs=epochs)

        # Populate initial search space.
        hp = self.oracle.get_space()
        dataset = fit_kwargs["x"]
        self._prepare_model_IO(hp, dataset=dataset)
        self.hypermodel.build(hp)
        self.oracle.update_space(hp)
        self.estimators = []
        self.estimator_results = {}
        self.on_search_begin()
        while True:
            trials = self.oracle.create_trials(self.parallelsim, self.tuner_id)
            # for trial in trials:
            #     print(trial.status)
            running_trials = []
            for trial in trials:
                if trial.status != trial_lib.TrialStatus.STOPPED:
                    running_trials.append(trial)
            if len(running_trials) == 0:
                break
            self.begin_trials(trials)
            self.run_trials(trials, epochs, dataset_idx, metadata, **fit_kwargs)
            self.end_trials(trials)
        self.on_search_end() 

        models = [est.create_model(self.estimator_results[est.getRunId()], est.getRunId(), metadata) for est in self.estimators]
        val_metrics = [self.estimator_results[est.getRunId()]['val_' + self.model_selection.evaluation_metric][-1] for est in self.estimators]
        best_model_idx = np.argmax(val_metrics) if self.oracle.objective.direction == "max"else np.argmin(val_metrics)
        best_model = models[best_model_idx]

        return ModelSelectionResult(best_model, self.estimator_results, models, [x+'__output' for x in self.model_selection.label_cols])

    def run_trials(self, trials, epochs, dataset_idx, metadata, **fit_kwargs):
        estimators = self.trials2estimators(trials, fit_kwargs["x"])
        ms = self.model_selection
        est_results = {model.getRunId():{'trial':trial} for trial, model in zip(trials, estimators)}

        for epoch in range(epochs):
            train_epoch = ms.backend.train_for_one_epoch(estimators, ms.store, dataset_idx, ms.feature_cols, ms.label_cols)
            update_model_results(est_results, train_epoch)

            val_epoch = ms.backend.train_for_one_epoch(estimators, ms.store, dataset_idx, ms.feature_cols, ms.label_cols, is_train=False)
            update_model_results(est_results, val_epoch)
            self.on_epoch_end(estimators=estimators, est_resutls=est_results, epoch=epoch)
        
        for est in estimators:
            self.estimators.append(est)
            self.estimator_results[est.getRunId()] = est_results[est.getRunId()]
        
    def on_epoch_end(self, estimators, est_resutls, epoch):
        
        for est in estimators:
            estimator_id = est.getRunId()
            trial = est_resutls[estimator_id]['trial']
            logs = {}
            for k in est_resutls[estimator_id]:
                if k is not 'trial':
                    logs[k] = est_resutls[estimator_id][k][-1]
            status = self.oracle.update_trial(
                trial.trial_id,
                metrics=logs,
                step=epoch
            )
            trial.status = status
            if trial.status == "STOPPED":
                est.getModel().stop_training = True