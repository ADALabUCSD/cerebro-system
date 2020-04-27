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

import os
from distutils.version import LooseVersion
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import numpy as np

np.random.seed(2020)


class _HP(object):
    def sample_value(self):
        """ randomly samples a value"""
        raise NotImplementedError()


class _HPChoice(_HP):
    def __init__(self, options):
        self.options = options

    def sample_value(self):
        return np.random.choice(self.options, 1)[0]


def hp_choice(options):
    """ categorical options
    :param options:
    :return:
    """

    if not type(options) == list:
        raise Exception('options has to be of type list.')

    return _HPChoice(options)


class _HPUniform(_HP):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def sample_value(self):
        return np.random.uniform(self.min, self.max, 1)[0]


def hp_uniform(min, max):
    """ uniform distribution bounded by min and max
    :param min:
    :param max:
    :return:
    """
    return _HPUniform(min, max)


class _HPQUniform(_HP):
    def __init__(self, min, max, q):
        self.min = min
        self.max = max
        self.q = q


def hp_quniform(min, max, q):
    """ quantized uniform distribution with a quantum of q, bounded by min and max
    :param min:
    :param max:
    :param q:
    :return:
    """
    return _HPQUniform(min, max, q)


class _HPLogUniform(_HP):
    def __init__(self, min, max):
        self.min = min
        self.max = max


def hp_loguniform(min, max):
    """ Log uniform distribution bounded by min and max
    :param min:
    :param max:
    :return:
    """
    return _HPLogUniform(min, max)


class _HPQLogUnifrom(_HP):
    def __init__(self, min, max, q):
        self.min = min
        self.max = max
        self.q = q


def hp_qloguniform(min, max, q):
    """ quantized log uniform distribution with a quantum of q, bounded by min and max
    :param min:
    :param max:
    :param q:
    :return:
    """
    return _HPQLogUnifrom(min, max, q)


class ModelSelection(object):
    """Cerebro model search base class"""

    def __init__(self, backend, store, validation, estimator_gen_fn, evaluation_metric, label_columns,
                 feature_columns, logdir, verbose):
        if is_valid_evaluation_metric(evaluation_metric):
            self.evaluation_metric = evaluation_metric
        else:
            raise Exception('Unsupported evaluation metric: {}'.format(evaluation_metric))

        self.backend = backend
        self.store = store
        self.validation = validation
        self.estimator_gen_fn = estimator_gen_fn
        self.label_cols = label_columns
        self.feature_cols = feature_columns
        self.logdir = logdir
        self.verbose = verbose

    def fit(self, df):
        """
        Trains ML models on the given data frame
        :param df:
        :return:
        """
        _, _, metadata, _ = self.backend.prepare_data(
            self.store, df, self.validation, label_columns=self.label_cols, feature_columns=self.feature_cols,
            compress_sparse=False, verbose=self.verbose, dataset_idx=None)  # no multiple datasets. Hence ids=None.

        # initialize backend and data loaders
        self.backend.initialize_workers()
        self.backend.initialize_data_loaders(self.store, None, self.feature_cols + self.label_cols)
        try:
            result = self._fit_on_prepared_data(None, metadata)
            return result
        finally:
            # teardown the backend workers
            self.backend.teardown_workers()

    def fit_on_prepared_data(self, dataset_index=None):
        """
        Trains ML models on already preapred data
        :return:
        """
        _, _, metadata, _ = self.backend.get_metadata_from_parquet(self.store, self.label_cols, self.feature_cols)

        # initialize backend and data loaders
        self.backend.initialize_workers()
        self.backend.initialize_data_loaders(self.store, dataset_index, self.feature_cols + self.label_cols)
        try:
            result = self._fit_on_prepared_data(dataset_index, metadata)
            return result
        finally:
            # teardown the backend workers
            self.backend.teardown_workers()

    def _fit_on_prepared_data(self):
        raise NotImplementedError('method not implemented')

    def _estimator_gen_fn_wrapper(self, params):
        # Disable GPUs when building the model to prevent memory leaks
        if LooseVersion(tf.__version__) >= LooseVersion('2.0.0'):
            # See https://github.com/tensorflow/tensorflow/issues/33168
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            tf.keras.set_session(tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})))

        est = self.estimator_gen_fn(params)
        est.setFeatureCols(self.feature_cols)
        est.setLabelCols(self.label_cols)
        est.setStore(self.store)
        est.setLogsDir(os.path.join(self.logdir, est.getRunId()))
        return est

    def _log_epoch_metrics_to_tensorboard(self, estimators, estimator_results):
        # logging to TensorBoard
        for est in estimators:
            log_model_epoch_metrics(os.path.join(self.logdir, est.getRunId()), estimator_results[est.getRunId()],
                                    est.getEpochs())

    def _log_hp_to_tensorboard(self, estimators, hparams):
        # logging to TensorBoard
        for i, est in enumerate(estimators):
            log_model_hps(os.path.join(self.logdir, est.getRunId()), est.getRunId(), hparams[i])


class ModelSelectionResult(object):
    """ModelSearchModel: Output of a ModelSearch fit() method"""

    def __init__(self, best_model, metrics, all_models, output_columns):
        self.best_model = best_model
        self.metrics = metrics
        self.all_models = all_models
        self.output_columns = output_columns

    def set_output_columns(self, output_columns):
        """
        Sets the output column names
        :param output_columns:
        """
        self.output_columns = output_columns
        self.best_model.setOutputCols(output_columns)
        for m in self.all_models:
            m.setOutputCols(output_columns)
        return self

    def keras(self):
        """
        Returns the best model in Keras format
        :return:
        """
        return self.best_model.keras()

    def transform(self, dataset):
        """
        Performs inference on a given dataset. Will run on CPU
        :param dataset:
        :return:
        """
        return self.best_model.transform(dataset)

    def get_best_model(self):
        """ Returns the best models
        :return: CerebroKerasModel
        """
        return self.best_model

    def get_history(self):
        return self.get_best_model_history()


    def get_best_model_history(self):
        """ Get best model training history
        :return:
        """
        return self.best_model.getHistory()

    def get_all_models(self):
        """ Returns a list of all models
        :return: list[CerebroKerasModel]
        """
        return self.all_models

    def get_all_model_history(self):
        """ Returns a list of model training metrics
        :return: list[Dict]
        """
        return self.metrics


def log_model_hps(logdir, model_id, hparams):
    """
    Logs model hyperparameters
    :param logdir:
    :param hparams:
    """
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams(hparams, trial_id=model_id)


def log_model_epoch_metrics(logdir, metrics, step_number):
    """
    Logs model epoch metrics
    :param logdir:
    :param metrics:
    :param step_number:
    """
    with tf.summary.create_file_writer(logdir).as_default():
        for key in metrics:
            tf.summary.scalar(key, metrics[key][step_number - 1], step=step_number)


def is_larger_better(metric_name):
    """
    Helper method to check whether high or low is better for a metric
    :param metric_name:
    :return:
    """
    if metric_name in ['r2']:
        return True
    else:
        return False


def is_valid_evaluation_metric(metric_name):
    """
    Helper method to check whether an evaluating metric is valid/supported.
    :param metric_name:
    :return:
    """
    if metric_name in ['loss', 'acc', 'accuracy']:
        return True
    else:
        return False


def update_model_results(estimator_results, epoch_results):
    """
    Method to update estimator results given epoch results
    :param estimator_results:
    :param epoch_results:
    """
    for model_id in epoch_results:
        res = epoch_results[model_id]
        for k in res:
            if k in estimator_results[model_id]:
                estimator_results[model_id][k].append(res[k])
            else:
                estimator_results[model_id][k] = [res[k]]
