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

import os
from distutils.version import LooseVersion
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import datetime
import numpy as np
from ..commons.util import fix_huggingface_layer_methods_and_add_to_custom_objects
from ..backend import constants

class _HP(object):
    def sample_value(self):
        """ randomly samples a value"""
        raise NotImplementedError()


class _HPChoice(_HP):
    def __init__(self, options):
        self.options = options
        self.rand = np.random.RandomState(constants.RANDOM_SEED)

    def sample_value(self):
        return self.rand.choice(self.options, 1)[0]


def hp_choice(options):
    """ Categorical options.

    :param options: List of options.
    """

    if not type(options) == list:
        raise Exception('options has to be of type list.')

    return _HPChoice(options)


class _HPUniform(_HP):
    def __init__(self, min, max):
        if min >= max:
            raise Exception('min should be smaller than max')

        self.min = min
        self.max = max

        self.rand = np.random.RandomState(constants.RANDOM_SEED)

    def sample_value(self):
        return self.rand.uniform(self.min, self.max, 1)[0]


def hp_uniform(min, max):
    """ Uniform distribution bounded by min and max.

    :param min: Minimum value
    :param max: Maximum value
    """
    return _HPUniform(min, max)


class _HPQUniform(_HP):
    def __init__(self, min, max, q):
        if min >= max:
            raise Exception('min should be smaller than max')

        if q >= (max - min):
            raise Exception('q should be smaller than (max-min)')
        self.min = min
        self.max = max
        self.q = q

        self.rand = np.random.RandomState(constants.RANDOM_SEED)

    def sample_value(self):
        t = round(self.rand.uniform(self.min, self.max, 1)[0] / self.q)
        return t * self.q


def hp_quniform(min, max, q):
    """ Quantized uniform distribution with a quantum of q, bounded by min and max. Returns a
     value like round(uniform(low, high) / q) * q.

    :param min: Minimum value
    :param max: Maximum value
    :param q: Quantum
    """
    return _HPQUniform(min, max, q)


class _HPLogUniform(_HP):
    def __init__(self, min, max):
        if min >= max:
            raise Exception('min should be smaller than max')

        self.min = min
        self.max = max
        self.rand = np.random.RandomState(constants.RANDOM_SEED)

    def sample_value(self):
        t = self.rand.uniform(self.min, self.max, 1)[0]
        return np.power(0.1, -t)


def hp_loguniform(min, max):
    """ Log uniform (base 10) distribution bounded by min and max.

    :param min: Exponent of the minimum value in base 10 (e.g., -4 for 0.0001).
    :param max: Exponent of the maximum value in based 10.
    """
    return _HPLogUniform(min, max)


class _HPQLogUnifrom(_HP):
    def __init__(self, min, max, q):
        if min >= max:
            raise Exception('min should be smaller than max')

        if q >= (max - min):
            raise Exception('q should be smaller than (max-min)')

        self.min = -min
        self.max = -max
        self.q = q

        self.rand = np.random.RandomState(constants.RANDOM_SEED)

    def sample_value(self):
        t = round(np.power(0.1, self.rand.uniform(self.min, self.max, 1)[0]) / self.q)
        return t * self.q


def hp_qloguniform(min, max, q):
    """ Quantized log uniform (base 10) distribution with a quantum of q, bounded by min and max. Returns a
     value like round(exp(uniform(low, high)) / q) * q.

    :param min: Exponent of the minimum value in base 10 (e.g., -4 for 0.0001).
    :param max: Exponent of the maximum value in base 10.
    :param q:   Quantum
    """
    return _HPQLogUnifrom(min, max, q)


class ModelSelection(object):
    """Cerebro model search base class"""

    def __init__(self, backend, store, validation, estimator_gen_fn, evaluation_metric, label_columns,
                 feature_columns, verbose):
        if is_valid_evaluation_metric(evaluation_metric):
            self.evaluation_metric = evaluation_metric
        else:
            raise Exception('Unsupported evaluation metric: {}'.format(evaluation_metric))

        self.backend = backend
        self.store = store
        self.store = store
        self.validation = validation
        self.estimator_gen_fn = estimator_gen_fn
        self.label_cols = label_columns
        self.feature_cols = feature_columns
        self.verbose = verbose

    def fit(self, df):
        """
        Execute the model selection/AutoML workload on the given DataFrame.

        :param df: Input DataFrame
        :return: cerebro.tune.ModelSelectionResult
        """
        if self.verbose >= 1: print(
            'CEREBRO => Time: {}, Preparing Data'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        _, _, metadata, _ = self.backend.prepare_data(self.store, df, self.validation)

        if self.verbose >= 1: print(
            'CEREBRO => Time: {}, Initializing Workers'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        # initialize backend and data loaders
        self.backend.initialize_workers()

        if self.verbose >= 1: print(
            'CEREBRO => Time: {}, Initializing Data Loaders'.format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.backend.initialize_data_loaders(self.store, self.feature_cols + self.label_cols)

        try:
            if self.verbose >= 1: print('CEREBRO => Time: {}, Launching Model Selection Workload'.format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            result = self._fit_on_prepared_data(metadata)
            return result
        finally:
            # teardown the backend workers
            if self.verbose >= 1: print(
                'CEREBRO => Time: {}, Terminating Workers'.format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            if self.backend is not None:
                self.backend.teardown_workers()

    def fit_on_prepared_data(self):
        """
         Execute the model selection/AutoML workload on already prepared data.

        :return: cerebro.tune.ModelSelectionResult
        """
        _, _, metadata, _ = self.backend.get_metadata_from_parquet(self.store, self.label_cols, self.feature_cols)

        # initialize backend and data loaders
        if self.verbose >= 1: print(
            'CEREBRO => Time: {}, Initializing Workers'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.backend.initialize_workers()

        if self.verbose >= 1: print(
            'CEREBRO => Time: {}, Initializing Data Loaders'.format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.backend.initialize_data_loaders(self.store, self.feature_cols + self.label_cols)

        try:
            if self.verbose >= 1: print('CEREBRO => Time: {}, Launching Model Selection Workload'.format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            result = self._fit_on_prepared_data(metadata)
            return result
        finally:
            # teardown the backend workers
            if self.verbose >= 1: print(
                'CEREBRO => Time: {}, Terminating Workers'.format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            if self.backend is not None:
                self.backend.teardown_workers()

    def _fit_on_prepared_data(self):
        raise NotImplementedError('method not implemented')

    def _estimator_gen_fn_wrapper(self, params):
        return estimator_gen_fn_wrapper(self.estimator_gen_fn, params, self.feature_cols, self.label_cols, self.store, self.verbose)

    def _log_epoch_metrics_to_tensorboard(self, estimators, estimator_results):
        # logging to TensorBoard
        log_epoch_metrics_to_tensorboard(estimators, estimator_results, self.store, self.verbose)

    def _log_hp_to_tensorboard(self, estimators, hparams):
        # logging to TensorBoard
        log_hp_to_tensorboard(estimators, hparams, self.store, self.verbose)


class ModelSelectionResult(object):
    """Output of a model selection workload performed using ``fit(df)`` or ``fit_on_prepared_data()`` method."""

    def __init__(self, best_model, metrics, all_models, output_columns):
        self.best_model = best_model
        self.metrics = metrics
        self.all_models = all_models
        self.output_columns = output_columns

    def set_output_columns(self, output_columns):
        """
        Sets the output column names.

        :param output_columns: Output column names.
        """
        self.output_columns = output_columns
        self.best_model.setOutputCols(output_columns)
        for m in self.all_models:
            m.setOutputCols(output_columns)
        return self

    def keras(self):
        """
        Returns the best model in Keras format.

        :return: TensorFlow Keras model
        """
        return self.best_model.keras()

    def transform(self, dataset):
        """
        Performs inference on a given dataset. Will run on CPU.

        :param dataset: Input DataFrame.
        :return: DataFrame
        """
        return self.best_model.transform(dataset)

    def get_best_model(self):
        """ Returns the best model.

        :return: :class:`cerebro.keras.CerebroModel`
        """
        return self.best_model

    def get_history(self):
        return self.get_best_model_history()

    def get_best_model_history(self):
        """ Get best model training history.

        :return: Dictionary containing all metric history at epoch granularity.
        """
        return self.best_model.getHistory()

    def get_all_models(self):
        """ Returns a list of all models.

        :return: List containing :class:`cerebro.keras.CerebroModel` objects
        """
        return self.all_models

    def get_all_model_history(self):
        """ Returns a list of model training metrics.

        :return: List of dictionaries, each containing all metric history at epoch granularity.
        """
        return self.metrics


def log_model_hps(logdir, model_id, hparams, verbose=1):
    """
    Logs model hyperparameters
    :param logdir:
    :param hparams:
    :param verbose:
    """
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams(hparams, trial_id=model_id)

    if verbose >= 1:
        print(
            ('CEREBRO => Time: {}, Model: {}, ' + ", ".join([k + ": {}" for k in hparams])).format(
                *([
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_id] + [
                      hparams[k] for k in hparams]
                  )
            )
        )


def log_model_epoch_metrics(model_id, logdir, metrics, step_number, verbose=1):
    """
    Logs model epoch metrics
    :param logdir:
    :param metrics:
    :param step_number:
    :param verbose:log_epoch_metrics_to_tensorboard(estimators, estimator_results, store, verbose)
    """
    with tf.summary.create_file_writer(logdir).as_default():
        for key in metrics:
            tf.summary.scalar(key, metrics[key][step_number - 1], step=step_number)

    if verbose >= 1:
        print(
            ('CEREBRO => Time: {}, Model: {}, Epoch: {}, ' + ", ".join([k + ": {}" for k in metrics])).format(
                *([
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_id, step_number] + [
                      metrics[k][step_number - 1] for k in metrics]
                  )
            )
        )


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

def log_hp_to_tensorboard(estimators, hparams, store, verbose):
    """
    Helper method to log hparams to the TensorBoard
    :param estimators:
    :param hparams:
    :param store: A single store common for all estimators or a dictionary containing stores for each estimator indexed by model id.
    :param verbose:
    """
    for i, est in enumerate(estimators):
        if type(store) == dict:
            a_store = store[est.getRunId()]
        else:
            a_store = store
        remote_store = a_store.to_remote(est.getRunId(), None)
        with remote_store.get_local_output_dir() as logs_dir:
            log_model_hps(logs_dir, est.getRunId(), hparams[i], verbose)
        remote_store.sync(logs_dir)

def log_epoch_metrics_to_tensorboard(estimators, estimator_results, store, verbose):
    """
    Helper method to log hparams to the TensorBoard
    :param estimators:
    :param estimator_results:
    :param store: A single store common for all estimators or a dictionary containing stores for each estimator indexed by model id.
    :param verbose:
    """
    for est in estimators:
        if type(store) == dict:
            a_store = store[est.getRunId()]
        else:
            a_store = store
        remote_store = a_store.to_remote(est.getRunId(), None)
        with remote_store.get_local_output_dir() as logs_dir:
            log_model_epoch_metrics(est.getRunId(), logs_dir, estimator_results[est.getRunId()], est.getEpochs(), verbose)
        remote_store.sync(logs_dir)


def estimator_gen_fn_wrapper(estimator_gen_fn, params, feature_cols, label_cols, store, verbose):
    """
    Function wrapping a user provided estimator gen function.
    """
    # Disable GPUs when building the model to prevent memory leaks
    if LooseVersion(tf.__version__) >= LooseVersion('2.0.0'):
        # See https://github.com/tensorflow/tensorflow/issues/33168
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        tf.keras.set_session(tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})))

    tf.keras.backend.clear_session()
    est = estimator_gen_fn(params)
    est.setHyperParams(params)
    est.setFeatureCols(feature_cols)
    est.setLabelCols(label_cols)
    est.setStore(store)
    est.setVerbose(verbose)

    # Workaround for the issue with huggingface layers needing a python
    # object as config (not a dict) and explicit definition of get_config method.
    # We monkey patch the __init__ method get_config methods of such layers.
    fix_huggingface_layer_methods_and_add_to_custom_objects(est)

    return est