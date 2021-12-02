from random import Random
import keras_tuner
from tensorflow._api.v2 import data
from .base import CerebroOracle
from ..sparktuner import SparkTuner
from keras_tuner.engine import trial as trial_lib, tuner
from ...tune.base import ModelSelection, update_model_results, ModelSelectionResult
from typing import Optional, List, Any, Dict, Union
import numpy as np
from enum import Enum

class Mode(Enum):
    EXPLOIT = 0
    EXPLORE = 1

class TrieNode(object):
    def __init__(self):
        super().__init__()
        self.num_leaves = 0
        self.children = {}
        self.hp_name = None

    def is_leaf(self):
        return len(self.children) == 0

class Trie(object):
    def __init__(self):
        super().__init__()
        self.root = TrieNode()

    def insert(self, hp_name):
        names = hp_name.split("/")

        new_word = False
        current_node = self.root
        nodes_on_path = [current_node]
        for name in names:
            if name not in current_node.children:
                current_node.children[name] = TrieNode()
                new_word = True
            current_node = current_node.children[name]
            nodes_on_path.append(current_node)
        current_node.hp_name = hp_name

        if new_word:
            for node in nodes_on_path:
                node.num_leaves += 1

    @property
    def nodes(self):
        return self._get_all_nodes(self.root)

    def _get_all_nodes(self, node):
        ret = [node]
        for key, value in node.children.items():
            ret += self._get_all_nodes(value)
        return ret

    def get_hp_names(self, node):
        if node.is_leaf():
            return [node.hp_name]
        ret = []
        for key, value in node.children.items():
            ret += self.get_hp_names(value)
        return ret


class GreedyOracle(CerebroOracle):
    def __init__(
        self, 
        objective, 
        exploration = 0.2,
        max_trials=None, 
        hyperparameters=None, 
        allow_new_entries=True, 
        tune_new_entries=True, 
        seed=None,
        initial_hps=None,
    ):
        self.exploration = exploration
        self.initial_hps = initial_hps or []
        self._tried_initial_hps = [False] * len(self.initial_hps)
        super().__init__(objective, max_trials=max_trials, hyperparameters=hyperparameters, allow_new_entries=allow_new_entries, tune_new_entries=tune_new_entries, seed=seed)
    
    def get_state(self):
        state = super().get_state()
        state.update(
            {
                "initial_hps": self.initial_hps,
                "tried_initial_hps": self._tried_initial_hps,
            }
        )
        return state

    def set_state(self, state):
        super().set_state(state)
        self.initial_hps = state["initial_hps"]
        self._tried_initial_hps = state["tried_initial_hps"]

    # Select a hyperparameter from the best trial
    def _select_hps(self):
        trie = Trie()
        best_hps = self._get_best_hps()
        for hp in best_hps.space:
            # Not picking the fixed hps for generating new values.
            if best_hps.is_active(hp) and not isinstance(
                hp, keras_tuner.engine.hyperparameters.Fixed
            ):
                trie.insert(hp.name)
        all_nodes = trie.nodes

        if len(all_nodes) <= 1:
            return []

        probabilities = np.array([1 / node.num_leaves for node in all_nodes])
        sum_p = np.sum(probabilities)
        probabilities = probabilities / sum_p
        node = np.random.choice(all_nodes, p=probabilities)

        return trie.get_hp_names(node)

    def _next_initial_hps(self):
        for index, hps in enumerate(self.initial_hps):
            if not self._tried_initial_hps[index]:
                self._tried_initial_hps[index] = True
                return hps

    def populate_space(self, trial_id):
        if not all(self._tried_initial_hps):
            values = self._next_initial_hps()
            return {
                "status": keras_tuner.engine.trial.TrialStatus.RUNNING,
                "values": values,
            }

        # Choose whether to explore new hps or exploit best hps
        mode = np.random.choice([Mode.EXPLOIT, Mode.EXPLORE], p=[1-self.exploration, self.exploration]) 
        if mode is Mode.EXPLOIT:
            for _ in range(self._max_collisions):
                hp_names = self._select_hps()
                values = self._generate_hp_values(hp_names)
                # Reached max collisions.
                if values is None:
                    continue
                # Values found.
                return {
                    "status": keras_tuner.engine.trial.TrialStatus.RUNNING,
                    "values": values,
                }
            # All stages reached max collisions.
            return {
                "status": keras_tuner.engine.trial.TrialStatus.STOPPED,
                "values": None,
            }
        else:
            values = self._random_values()
            if values is None:
                return {"status": trial_lib.TrialStatus.STOPPED, "values": None}
            return {"status": trial_lib.TrialStatus.RUNNING, "values": values}

    def _get_best_hps(self):
        best_trials = self.get_best_trials()
        if best_trials:
            return best_trials[0].hyperparameters.copy()
        else:
            return self.hyperparameters.copy()
    
    # Generate hps by randomize on the given hp
    def _generate_hp_values(self, hp_names):
        best_hps = self._get_best_hps()

        collisions = 0
        while True:
            hps = keras_tuner.HyperParameters()
            # Generate a set of random values.
            for hp in self.hyperparameters.space:
                hps.merge([hp])
                # if not active, do nothing.
                # if active, check if selected to be changed.
                if hps.is_active(hp):
                    # if was active and not selected, do nothing.
                    if best_hps.is_active(hp.name) and hp.name not in hp_names:
                        hps.values[hp.name] = best_hps.values[hp.name]
                        continue
                    # if was not active or selected, sample.
                    hps.values[hp.name] = hp.random_sample(self._seed_state)
                    self._seed_state += 1
            values = hps.values
            # Keep trying until the set of values is unique,
            # or until we exit due to too many collisions.
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                collisions += 1
                if collisions <= self._max_collisions:
                    continue
                return None
            self._tried_so_far.add(values_hash)
            break
        return values
    
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

    # Cold star generates n random profiles as the initial pool
    def cold_start(self, n):
        for i in range(n):
            values = self._random_values()
            if values is None:
                break
            else:
                self.initial_hps.append(values)
        self._tried_initial_hps = [False] * len(self.initial_hps)

class GreedySearch(SparkTuner):
    def __init__(
        self, 
        hypermodel, 
        parallelism,
        model_selection: ModelSelection, 
        exploration: Union[int, float] = None,
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
        if exploration:
            if type(exploration) is float:
                n = exploration
            else:
                n = float(exploration / parallelism)
        else:
            n = 0.2
        oracle = GreedyOracle(
            objective=objective,
            exploration=n,
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
        cold_start=5,
        **fit_kwargs
    ):
        self._display.verbose = verbose
        self.hypermodel.hypermodel.set_fit_args(validation_split, epochs=epochs)

        # Populate initial search space.
        hp = self.oracle.get_space()
        dataset = fit_kwargs["x"]
        self._prepare_model_IO(hp, dataset=dataset)
        self.hypermodel.build(hp)
        self.oracle.update_space(hp)
        
        # Generate initial hps for cold start
        self.oracle.cold_start(cold_start)

        self.estimators = []
        self.estimator_results = {}
        self.on_search_begin()
        while True:
            trials = self.oracle.create_trials(self.parallelsim, self.tuner_id)
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
