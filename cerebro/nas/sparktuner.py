from enum import auto
from autokeras.auto_model import AutoModel
from keras_tuner.engine.tuner import Tuner
import tensorflow as tf
import keras_tuner as kt
from tensorflow._api.v2 import data

from tensorflow.keras import callbacks as tf_callbacks
from autokeras.utils import utils, data_utils
from autokeras.engine.tuner import AutoTuner
from ..tune.base import ModelSelection


class SparkTuner(kt.engine.tuner.Tuner):
    """
    SparkTuner inherits AutoTuner to tune the preprocessor blocks. Also it over-writes run_trial method to using cerebro as the underling training system

    [param] - [backend, store, validation, evaluation_metric, label_column, feature_column, verbose]: For construction of modelsection object

    param - oracle: Tuning kernel
    param - hypermodel: Hypermodel which implements build method, hypermodel.build(hp) will give a keras model
    """

    def __init__(
            self,
            oracle,
            hypermodel,
            model_selection: ModelSelection,
            **kwargs):
        self._finished = False
        self.model_selection = model_selection
        super().__init__(oracle, hypermodel, **kwargs)

    def _populate_initial_space(self):
        return

    def _prepare_model_IO(self, hp, dataset):
        """
        Prepare for building the Keras model.
        Set the input shapes and output shapes of the HyperModel
        """
        self.hypermodel.hypermodel.set_io_shapes(data_utils.dataset_shape(dataset))

    """
    Over-write this function to train one epoch using cerebro

    kwargs["x"] is a tf.data.dataset containing train_x and train_y
    """
    def _build_and_fit_model(self, trial, *args, **kwargs):
        dataset = kwargs["x"]
        self._prepare_model_IO(trial.hyperparameters, dataset=dataset)
        model = self.hypermodel.build(trial.hyperparameters)
        self.adapt(model, dataset)
        params = {
            'model': model,
            'optimizer': model.optimizer, # keras opt not str
            'loss': self.hypermodel._get_loss(), # not sure
            'metrics': self.hypermodel._get_metrics(),
            'bs': self.hypermodel.batch_size
        }
        _, history = self.spark_fit(
            params, **kwargs
        )
        return history

    """
    Train a generated model with params as hyperparameter
    The model is wrapped with spark estimator and is trained for one epoch.

    params: normal training hyperparameter to construct the estimator
    kwargs['x']: tf.data.dataset.zip(train_x, train_y)
    kwargs['validation_data']: 
    """
    def spark_fit(self, params, **kwargs):
        ms = self.model_selection 
        est = ms._estimator_gen_fn_wrapper(params)
        #TODO Log to tensorboard
        epoch_rel = ms.backend.train_for_one_epoch(est, ms.store, ms.feature_cols, ms.label_cols)
        hist = 0
        for k in epoch_rel:
            hist = hist + epoch_rel[k]
        return hist


    def search(
            self,
            epochs=None,
            callbacks=None,
            validation_split=0,
            verbose=1,
            **fit_kwargs
        ):
            """Search for the best HyperParameters.

            If there is not early-stopping in the callbacks, the early-stopping callback
            is injected to accelerate the search process. At the end of the search, the
            best model will be fully trained with the specified number of epochs.

            # Arguments
                callbacks: A list of callback functions. Defaults to None.
                validation_split: Float.
            """
            if self._finished:
                return

            if callbacks is None:
                callbacks = []

            self.hypermodel.hypermodel.set_fit_args(validation_split, epochs=epochs)

            # Insert early-stopping for adaptive number of epochs.
            epochs_provided = True
            if epochs is None:
                epochs_provided = False
                epochs = 1000
                if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
                    callbacks.append(
                        tf_callbacks.EarlyStopping(patience=10, min_delta=1e-4)
                    )

            # Insert early-stopping for acceleration.
            early_stopping_inserted = False
            new_callbacks = self._deepcopy_callbacks(callbacks)
            if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
                early_stopping_inserted = True
                new_callbacks.append(
                    tf_callbacks.EarlyStopping(patience=10, min_delta=1e-4)
                )
            # Populate initial search space.
            hp = self.oracle.get_space()
            dataset = fit_kwargs["x"]
            self._prepare_model_IO(hp, dataset=dataset)
            self.hypermodel.build(hp)
            self.oracle.update_space(hp)
            self.oracle.init_search_space()

            super().search(
                epochs=epochs, callbacks=new_callbacks, verbose=verbose, **fit_kwargs
            )

    def space_initialize_test(
            self,
            validation_split,
            epochs,
            **fit_kwargs
        ):
            self.hypermodel.hypermodel.set_fit_args(validation_split, epochs=epochs)

            # Populate initial search space.
            hp = self.oracle.get_space()
            dataset = fit_kwargs["x"]
            self._prepare_model_IO(hp, dataset=dataset)
            self.hypermodel.build(hp)
            self.oracle.update_space(hp)