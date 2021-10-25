from enum import auto
from autokeras.auto_model import AutoModel
from keras_tuner.engine.hypermodel import HyperModel
from keras_tuner.engine.tuner import Tuner
import tensorflow as tf
import keras_tuner as kt

from autokeras.utils import utils, data_utils
from autokeras.engine.tuner import AutoTuner
from keras import estimator
from keras.spark.estimator import SparkEstimator
from tune.base import ModelSelection


class SparkTuner(AutoTuner):
    """
    SparkTuner inherits AutoTuner to tune the preprocessor blocks. Also it over-writes run_trial method to using cerebro as the underling training system

    [param] - [backend, store, validation, evaluation_metric, label_column, feature_column, verbose]: For construction of modelsection object

    param - oracle: Tuning kernel
    param - hypermodel: Hypermodel which implements build method, hypermodel.build(hp) will give a keras model
    """

    def __init__(
            self,
            oracle,
            hypermodel: HyperModel,
            backend, store, validation, evaluation_metric, label_columns, feature_columns, verbose,
            **kwargs):

        """
        Default estimator_generation_function used to initialize model_selection class, all parameters are extracted from params
        """
        def est_gen_fun(params):
            model = params['model']
            opt = params['optimizer']
            loss = params['loss']
            metrics = params['metrics']
            bs = params['batch_size']
            estimator = SparkEstimator(
                model=model,
                optimizer=opt,
                loss=loss,
                metrics=metrics,
                batch_size=bs
            )
            return estimator

        """
        We temporarily keep the entire ModelSelection class here for compatibility and potential usage of its methods
        It encapsulates backend engine which shall be used in the training process
        """
        self.ms = ModelSelection(
            backend,
            store,
            validation,
            est_gen_fun,
            evaluation_metric,
            label_columns,
            feature_columns,
            verbose
        )

        super().__init__(oracle, hypermodel, **kwargs)

    """
    Over-write this function to train one epoch using cerebro
    """
    def _build_and_fit_model(self, trial, *args, **kwargs):
        model = self.hypermodel.build(trial.hyperparameters)
        (
            pipeline,
            kwargs["x"],
            kwargs["validation_data"],
        ) = self._prepare_model_build(trial.hyperparameters, **kwargs)
        pipeline.save(self._pipeline_path(trial.trial_id))

        self.adapt(model, kwargs["x"])

        _, history = self.spark_fit(
            model, self.hypermodel.batch_size, **kwargs
        )
        return history

    # TODO
    def spark_fit(self, model, batch_size, **kwargs):
        pass