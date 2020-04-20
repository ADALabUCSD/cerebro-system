# Copyright 2020 University of California Regents. All Rights Reserved.
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

from .base import ModelSearch, _HP, _HPChoice, _HPUniform, update_model_results, is_larger_better, ModelSearchModel
from hyperopt import tpe, hp, Trials, STATUS_OK, STATUS_RUNNING
from hyperopt.base import Domain
import numpy as np


def _validate_and_generate_hyperopt_search_space(search_space):
    if not type(search_space) == dict:
        raise Exception('Search space has to be type dict. Provided: {}'.format(type(search_space)))

    if not all([isinstance(k, str) for k in search_space.keys()]):
        raise Exception('Only string values are allowed for hyperparameter space keys.')

    if not all([isinstance(k, _HP) for k in search_space.values()]):
        raise Exception('All hyperparameter space values has to be of type cerebro.tune.base._HP.'
                        ' Nested search spaces are not supported yet')

    hyperopt_space = {}
    for k in search_space:
        if isinstance(search_space[k], _HPChoice):
            hyperopt_space[k] = hp.choice(k, search_space[k].options)
        elif isinstance(search_space[k], _HPUniform):
            hyperopt_space[k] = hp.uniform(k, search_space[k].min, search_space[k].max)
        else:
            raise Exception('Unsupported hyperparameter option type: {}'.format(type(search_space[k])))

    return hyperopt_space


class HyperOpt(ModelSearch):
    """Performs HyperOpt[https://github.com/hyperopt/hyperopt] search using the given param grid"""

    def __init__(self, backend, store, estimator_gen_fn, search_space, num_params, num_epochs, validation=0.25,
                 evaluation_metric='loss', label_column='label', feature_column='features', logdir='./logs',
                 parallelism=None, verbose=2):
        super(HyperOpt, self).__init__(backend, store, validation, estimator_gen_fn, evaluation_metric,
                                       label_column, feature_column, logdir, verbose)

        if is_larger_better(evaluation_metric):
            raise Exception('HyperOpt supports only minimizing evaluation metrics (e.g., loss)')

        self.search_space = search_space
        # validate the search space
        self.hyperopt_search_space = _validate_and_generate_hyperopt_search_space(search_space)

        if parallelism is None:
            parallelism = backend.num_processes()
        self.parallelism = parallelism
        self.num_params = num_params
        self.num_epochs = num_epochs

    def _fit_on_prepared_data(self, dataset_idx, metadata):
        trials = Trials()
        domain = Domain(None, self.hyperopt_search_space)

        all_estimators = []
        all_estimator_results = {}
        for i in range(0, self.num_params, self.parallelism):
            n = min(self.num_params - i, self.parallelism)

            # Using HyperOpt TPE to generate parameters
            hyperopt_params = []
            for j in range(i, i + n):
                new_param = tpe.suggest([j], domain, trials, np.random.randint(0, 2 ** 31 - 1))
                new_param[0]['status'] = STATUS_RUNNING

                trials.insert_trial_docs(new_param)
                trials.refresh()
                hyperopt_params.append(new_param[0])

            # Generating Cerebro params from HyperOpt params
            estimator_param_maps = []
            for hyperopt_param in hyperopt_params:
                param = {}
                for k in hyperopt_param['misc']['vals']:
                    val = hyperopt_param['misc']['vals'][k][0].item()
                    if isinstance(self.search_space[k], _HPChoice):
                        # if the hyperparamer is a choice the index is returned
                        val = self.search_space[k].options[val]
                    param[k] = val
                estimator_param_maps.append(param)

            # Generating Cerebro estimators
            estimators = [self._estimator_gen_fn_wrapper(param) for param in estimator_param_maps]
            estimator_results = {model.getRunId(): {} for model in estimators}
            # log hyperparameters to TensorBoard
            self._log_hp_to_tensorboard(estimators, estimator_param_maps)

            # Trains the models up to the number of epochs specified. For each iteration also performs validation
            for epoch in range(self.num_epochs):
                epoch_results = self.backend.train_for_one_epoch(estimators, self.store, dataset_idx, self.feature_col,
                                                                 self.label_col)
                update_model_results(estimator_results, epoch_results)

                epoch_results = self.backend.train_for_one_epoch(estimators, self.store, dataset_idx, self.feature_col,
                                                                 self.label_col, is_train=False)
                update_model_results(estimator_results, epoch_results)

                self._log_epoch_metrics_to_tensorboard(estimators, estimator_results)

            all_estimators.extend(estimators)
            all_estimator_results.update(estimator_results)

            # HyperOpt TPE update
            for i, hyperopt_param in enumerate(hyperopt_params):
                hyperopt_param['status'] = STATUS_OK
                hyperopt_param['result'] = {'loss': estimator_results[estimators[i].getRunId()][
                    'val_' + self.evaluation_metric][-1], 'status': STATUS_OK}
            trials.refresh()

        # find the best model and crate ModelSearchModel
        models = [est.create_model(all_estimator_results[est.getRunId()], est.getRunId(), metadata) for est in
                  all_estimators]
        val_metrics = [all_estimator_results[est.getRunId()]['val_' + self.evaluation_metric][-1] for est in
                       all_estimators]
        best_model = models[np.argmin(val_metrics)]

        return ModelSearchModel(best_model, estimator_results, models)
