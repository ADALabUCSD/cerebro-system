from random import Random
import keras_tuner
from tensorflow._api.v2 import data
from .base import CerebroOracle
from ..sparktuner import SparkTuner
from keras_tuner.engine import trial as trial_lib, tuner
from ...tune.base import ModelSelection, update_model_results, ModelSelectionResult
from typing import Optional, List, Any, Dict, Union
import numpy as np


class TrieNode(object):
    def __init__(self):
        super().__init__()
        self.num_child = 0
        self.children = {}
        self.hp_name = None

    def is_leaf(self):
        return len(self.children) is 0

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
        exploration = 1,
        max_trials=None, 
        hyperparameters=None, 
        allow_new_entries=True, 
        tune_new_entries=True, 
        seed=None
    ):
        self.exploration = exploration
        super().__init__(objective, max_trials=max_trials, hyperparameters=hyperparameters, allow_new_entries=allow_new_entries, tune_new_entries=tune_new_entries, seed=seed)

    def select_hps(self):
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

        return trie

    def _get_best_hps(self):
        best_trials = self.get_best_trials()
        if best_trials:
            return best_trials[0].hyperparameters.copy()
        else:
            return self.hyperparameters.copy()

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
            if type(exploration) is int:
                n = exploration
            else:
                n = int(parallelism * exploration)
        else:
            n = 1
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