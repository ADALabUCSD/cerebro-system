# Copyright 2020 Supun Nakandala, Yuhao Zhang, and Arun Kumar. All Rights Reserved.
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

import itertools
from sqlalchemy import and_
import numpy as np

from .base import ModelSelection, is_larger_better, ModelSelectionResult, _HP, _HPChoice, update_model_results
from ..db.dao import Model, Metric
from ..commons.constants import CREATED_STATUS, RUNNING_STATUS, COMPLETED_STATUS


class GridSearch(ModelSelection):
    """Performs grid search using the given param grid

    :param backend: Cerebro backend object (e.g., SparkBackend).
    :param store: Cerebro store object (e.g., LocalStore, HDFSStore).
    :param estimator_gen_fn: A function which takes
     in a dictionary of parameters and returns a Cerebro Estimator (e.g., cerebro.SparkEstimator).
    :param search_space: A dictionary object defining the parameter search space.
    :param num_epochs: Number of maximum epochs each model should be trained for.
    :param evaluation_metric: Evaluation metric used to pick the best model (default "loss").
    :param validation: (Optional) The ratio of the validation set (default: 0.25) or a string defining the column name
     defining the validation set. In the latter case the column value can be bool or int.
    :param label_columns: (Optional) A list containing the names of the label/output columns (default ['label']).
    :param feature_columns: (Optional) A list containing the names of the feature columns (default ['features']).
    :param verbose: Debug output verbosity (0-2). Defaults to 1.

    :return: :class:`cerebro.tune.ModelSelectionResult`
    """

    def __init__(self, backend, store, estimator_gen_fn, search_space, num_epochs,
                 evaluation_metric='loss', validation=0.25, label_columns=['label'], feature_columns=['features'],
                 verbose=1):
        super(GridSearch, self).__init__(backend, store, validation, estimator_gen_fn, evaluation_metric,
                                         label_columns, feature_columns, verbose)

        self.search_space = search_space
        # validate the search space
        self._validate_search_space()

        self.estimator_param_maps = self._generate_all_param_maps()
        self.num_epochs = num_epochs

    def _validate_search_space(self):
        search_space = self.search_space
        if not type(search_space) == dict:
            raise Exception('Search space has to be type dict. Provided: {}'.format(type(search_space)))

        if not all([isinstance(k, str) for k in search_space.keys()]):
            raise Exception('Only string values are allowed for hyperparameter space keys.')

        if not all([isinstance(k, _HPChoice) for k in search_space.values()]):
            raise Exception('All hyperparameter space values has to be of type cerebro.tune.base._HPChoice.'
                            ' Nested search spaces are not supported yet')

    def _generate_all_param_maps(self):
        keys = self.search_space.keys()
        grid_values = [v.options for v in self.search_space.values()]

        def _to_key_value_pairs(keys, values):
            # values = [v if isinstance(v, list) else v() for v in values]
            return [(key, value) for key, value in zip(keys, values)]

        return [dict(_to_key_value_pairs(keys, prod)) for prod in itertools.product(*[v if isinstance(v, list) else \
                                                                                          v() for v in grid_values])]

    def _fit_on_prepared_data(self, metadata):
        return _fit_on_prepared_data(self, metadata)


class HILGridSearch(GridSearch):
    """Performs intermittent HIL grid search using the given param grid
    :param exp_id: Experiment ID.
    :param backend: Cerebro backend object (e.g., SparkBackend).
    :param store: Cerebro store object (e.g., LocalStore, HDFSStore).
    :param estimator_gen_fn: A function which takes 
     in a dictionary of parameters and returns a Cerebro Estimator (e.g., cerebro.SparkEstimator).
    :param search_space: A dictionary object defining the parameter search space.
    :param num_epochs: Number of maximum epochs each model should be trained for.
    :param db: SQLAlchemy DB object.
    :param label_columns: (Optional) A list containing the names of the label/output columns (default ['label']).
    :param feature_columns: (Optional) A list containing the names of the feature columns (default ['features']).
    :param verbose: Debug output verbosity (0-2). Defaults to 1.
    """

    def __init__(self, exp_id, backend, store, estimator_gen_fn, search_space, num_epochs, db,
                 label_columns=['label'], feature_columns=['features'], verbose=1):
        super(HILGridSearch, self).__init__(backend=backend, store=store, estimator_gen_fn=estimator_gen_fn, search_space=search_space,
            num_epochs=num_epochs, label_columns=label_columns, feature_columns=feature_columns, verbose=verbose)
        self.exp_id = exp_id
        self.db = db

    def _fit_on_prepared_data(self, metadata):
        return _hil_fit_on_prepared_data(self, metadata)


class RandomSearch(ModelSelection):
    """ Performs Random Search over the param grid

    :param backend: Cerebro backend object (e.g., SparkBackend).
    :param store: Cerebro store object (e.g., LocalStore, HDFSStore).
    :param estimator_gen_fn: A function which takes
     in a dictionary of parameters and returns a Cerebro Estimator (e.g., cerebro.SparkEstimator).
    :param search_space: A dictionary object defining the parameter search space.
    :param num_models: Maximum number of models to be explored.
    :param num_epochs: Number of maximum epochs each model should be trained for.
    :param evaluation_metric: Evaluation metric used to pick the best model (default: "loss").
    :param validation: (Optional) The ratio of the validation set (default: 0.25) or a string defining the column name
     defining the validation set. In the latter case the column value can be bool or int.
    :param label_columns: (Optional) A list containing the names of the label/output columns (default ['label']).
    :param feature_columns: (Optional) A list containing the names of the feature columns (default ['features']).
    :param verbose: Debug output verbosity (0-2). Defaults to 1.

    :return: :class:`cerebro.tune.ModelSelectionResult`
    """

    def __init__(self, backend, store, estimator_gen_fn, search_space, num_models, num_epochs, evaluation_metric='loss',
                 validation=0.25, label_columns=['label'], feature_columns=['features'], verbose=1):
        super(RandomSearch, self).__init__(backend, store, validation, estimator_gen_fn, evaluation_metric,
                                           label_columns, feature_columns, verbose)

        self.search_space = search_space
        # validate the search space
        self._validate_search_space()

        self.num_params = num_models
        self.estimator_param_maps = self._generate_all_param_maps()
        self.num_epochs = num_epochs

    def _validate_search_space(self):
        search_space = self.search_space
        if not type(search_space) == dict:
            raise Exception('Search space has to be type dict. Provided: {}'.format(type(search_space)))

        if not all([isinstance(k, str) for k in search_space.keys()]):
            raise Exception('Only string values are allowed for hyperparameter space keys.')

        if not all([isinstance(k, _HP) for k in search_space.values()]):
            raise Exception('All hyperparameter space values has to be of type cerebro.tune.base._HP.'
                            ' Nested search spaces are not supported yet')

    def _generate_all_param_maps(self):
        params = []
        keys = self.search_space.keys()
        for _ in range(self.num_params):
            param_dict = {}
            for k in keys:
                param_dict[k] = self.search_space[k].sample_value()
            params.append(param_dict)
        return params

    def _fit_on_prepared_data(self, metadata):
        return _fit_on_prepared_data(self, metadata)


class HILRandomSearch(RandomSearch):
    """Performs intermittent HIL random search using the given param grid
    :param exp_id: Experiment ID.
    :param backend: Cerebro backend object (e.g., SparkBackend).
    :param store: Cerebro store object (e.g., LocalStore, HDFSStore).
    :param estimator_gen_fn: A function which takes 
     in a dictionary of parameters and returns a Cerebro Estimator (e.g., cerebro.SparkEstimator).
    :param search_space: A dictionary object defining the parameter search space.
    :param num_models: Maximum number of models to be explored.
    :param num_epochs: Number of maximum epochs each model should be trained for.
    :param db: SQLAlchemy DB object.
    :param label_columns: (Optional) A list containing the names of the label/output columns (default ['label']).
    :param feature_columns: (Optional) A list containing the names of the feature columns (default ['features']).
    :param verbose: Debug output verbosity (0-2). Defaults to 1.

    """

    def __init__(self, exp_id, backend, store, estimator_gen_fn, search_space, num_models, num_epochs, db,
                 label_columns=['label'], feature_columns=['features'], verbose=1):
        super(HILRandomSearch, self).__init__(backend=backend, store=store, estimator_gen_fn=estimator_gen_fn, search_space=search_space,
            num_models=num_models, num_epochs=num_epochs, label_columns=label_columns, feature_columns=feature_columns, verbose=verbose)
        self.exp_id = exp_id
        self.db = db

    def _fit_on_prepared_data(self, metadata):
        return _hil_fit_on_prepared_data(self, metadata)


# Batch implementation (i.e., without any user interaction) of model selection.
def _fit_on_prepared_data(self, metadata):
    # create estimators
    estimators = [self._estimator_gen_fn_wrapper(param) for param in self.estimator_param_maps]
    estimator_results = {model.getRunId(): {} for model in estimators}

    # log hyperparameters to TensorBoard
    self._log_hp_to_tensorboard(estimators, self.estimator_param_maps)

    # Trains the models up to the number of epochs specified. For each iteration also performs validation
    for epoch in range(self.num_epochs):
        epoch_results = self.backend.train_for_one_epoch(estimators, self.store, self.feature_cols,
                                                         self.label_cols)
        update_model_results(estimator_results, epoch_results)

        epoch_results = self.backend.train_for_one_epoch(estimators, self.store, self.feature_cols,
                                                         self.label_cols, is_train=False)
        update_model_results(estimator_results, epoch_results)

        self._log_epoch_metrics_to_tensorboard(estimators, estimator_results)

    # find the best model and crate ModelSearchModel
    models = [est.create_model(estimator_results[est.getRunId()], est.getRunId(), metadata) for est in estimators]
    val_metrics = [estimator_results[est.getRunId()]['val_' + self.evaluation_metric][-1] for est in estimators]
    best_model_idx = np.argmax(val_metrics) if is_larger_better(self.evaluation_metric) else np.argmin(val_metrics)
    best_model = models[best_model_idx]

    return ModelSelectionResult(best_model, estimator_results, models, [x+'__output' for x in self.label_cols])


# Human-in-the-loop implementation
def _hil_fit_on_prepared_data(self, metadata):
    exp_id = self.exp_id
    db = self.db
    max_epochs = self.num_epochs

    # create estimators
    all_models = Model.query.filter(and_(Model.status.in_([CREATED_STATUS, RUNNING_STATUS])), Model.max_train_epochs > Model.num_trained_epochs).all()
    while all_models is not None and len(all_models) > 0:
        estimators = []
        estimator_results = {}
        for m in all_models:
            param = {}
            for d in m.param_vals:
                #TODO handle dtype
                param[d.name] = d.value
            est = self._estimator_gen_fn_wrapper(param)
            est.setRunId(m.id)
            estimators.append(est)

            if m.status == CREATED_STATUS:
                m.status = RUNNING_STATUS
                # log hyperparameters to TensorBoard
                self._log_hp_to_tensorboard([est], [param])
            
            estimator_results[m.id] = {}
            for metric in m.metrics:
                estimator_results[m.id][metric.name] = [float(x) for x in metric.values.split(',')]
        db.session.commit()

        # Trains all the models for one epoch. Also performs validation
        epoch_results = self.backend.train_for_one_epoch(estimators, self.store, self.feature_cols, self.label_cols)
        update_model_results(estimator_results, epoch_results)

        epoch_results = self.backend.train_for_one_epoch(estimators, self.store, self.feature_cols, self.label_cols, is_train=False)
        update_model_results(estimator_results, epoch_results)

        self._log_epoch_metrics_to_tensorboard(estimators, estimator_results)

        for m in all_models:
            est_results = estimator_results[m.id]
            metrics = m.metrics.all()
            if len(metrics) == 0:
                for k in est_results:
                    db.session.add(Metric(m.id, k, est_results[k]))
            else:
                for k in est_results:
                    metric = [metric for metric in metrics if metric.name == k][0]
                    metric.values = ",".join(["{:.4f}".format(x) for x in est_results[k]])

        for m in all_models:
            m.num_trained_epochs += 1
            if m.num_trained_epochs == m.max_train_epochs:
                m.status = COMPLETED_STATUS
        db.session.commit()

        # create estimators for the next epoch
        all_models = Model.query.filter(and_(Model.status.in_([CREATED_STATUS, RUNNING_STATUS])), Model.max_train_epochs > Model.num_trained_epochs).all()
