from enum import auto
from math import e
from autokeras.auto_model import AutoModel
from keras_tuner.engine.tuner import Tuner
from pyspark.ml.param import Params
import tensorflow as tf
import keras_tuner as kt
from tensorflow._api.v2 import data
from tensorflow.python.util import nest
from tensorflow.keras.layers.experimental import preprocessing
import collections

from tensorflow.keras import callbacks as tf_callbacks
from autokeras.utils import utils, data_utils
from autokeras.engine.tuner import AutoTuner
from autokeras import keras_layers

from ..tune.base import ModelSelection, update_model_results
from keras_tuner.engine import trial as trial_lib


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
            parallelism,
            model_selection: ModelSelection,
            **kwargs):
        self._finished = False
        self.model_selection = model_selection
        self.parallelsim = parallelism
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
        hm = self.hypermodel.hypermodel
        self._prepare_model_IO(trial.hyperparameters, dataset=dataset)
        model = self.hypermodel.build(trial.hyperparameters)
        self.adapt(model, dataset)
        params = {
            'model': model,
            'optimizer': model.optimizer, # keras opt not str
            'loss': self.hypermodel._get_loss(), # not sure
            'metrics': self.hypermodel._get_metrics(),
            'batch_size': self.hypermodel.batch_size
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
            validation_split=0.2,
            verbose=1,
            dataset_idx=None,
            metadata=None,
            **fit_kwargs
        ):
            """Search for the best HyperParameters.

            # Arguments
                callbacks: A list of callback functions. Defaults to None.
                validation_split: Float.
            """
            raise NotImplementedError
            # if self._finished:
            #     return

            # if callbacks is None:
            #     callbacks = []

            # self.hypermodel.hypermodel.set_fit_args(validation_split, epochs=epochs)

            # # Insert early-stopping for adaptive number of epochs.
            # epochs_provided = True
            # if epochs is None:
            #     epochs_provided = False
            #     epochs = 1000
            #     if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
            #         callbacks.append(
            #             tf_callbacks.EarlyStopping(patience=10, min_delta=1e-4)
            #         )

            # # Insert early-stopping for acceleration.
            # early_stopping_inserted = False
            # new_callbacks = self._deepcopy_callbacks(callbacks)
            # if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
            #     early_stopping_inserted = True
            #     new_callbacks.append(
            #         tf_callbacks.EarlyStopping(patience=10, min_delta=1e-4)
            #     )
            # # Populate initial search space.
            # hp = self.oracle.get_space()
            # dataset = fit_kwargs["x"]
            # self._prepare_model_IO(hp, dataset=dataset)
            # self.hypermodel.build(hp)
            # self.oracle.update_space(hp)
            # self.oracle.init_search_space()

            # # super().search(
            # #     epochs=epochs, callbacks=new_callbacks, verbose=verbose, **fit_kwargs
            # # )
            # self.on_search_begin()
            # while True:
            #     trials = self.oracle.create_trials(self.parallelsim)
            #     running_trials = []
            #     for trial in trials:
            #         if trial.status != trial_lib.TrialStatus.STOPPED:
            #             running_trials.append(trial)
            #     if len(running_trials) == 0:
            #         break
            #     self.begin_trials(trials)
            #     self.run_trials(trials, epochs, **fit_kwargs)
            #     self.end_trials(trials)

    def begin_trials(self, trials):
        for trial in trials:
            super().on_trial_begin(trial)

    def end_trials(self, trials):
        for trial in trials:
            super().on_trial_end(trial)
    
    def fixed_arch_search(
        self, 
        hp, 
        validation_split=0.2, 
        data_idx=None,
        metadata=None,
        verbose=1,
        **kwargs,
    ):
        epochs = kwargs['epoch'] or 10
        self._display.verbose = verbose
        self.hypermodel.hypermodel.set_fit_args(validation_split, epochs=epochs)

        # Populate initial search space.
        hp = self.oracle.get_space()
        dataset = kwargs["x"]
        self._prepare_model_IO(hp, dataset=dataset)
        self.hypermodel.build(hp)
        self.oracle.update_space(hp)

        i = 0
        ms = self.model_selection
        est_results = {}
        while i < self.oracle.max_trials:
            print(i)
            trials = self.oracle.create_trials(1, self.tuner_id)
            trial = trials[0]
            for opt in hp._hps['optimizer'][0].values:
                for lr in hp._hps['learning_rate'][0].values:
                    for bs in hp._hps['batch_size'][0].values:
                        if i < self.oracle.max_trials:
                            estimator = self.trial_from_config_to_est(
                                trial=trial,
                                dataset=dataset,
                                learning_rate=lr,
                                batch_size=bs,
                                optimizer=opt
                            )
                            est_results[estimator.getRunId()] = {}
                            est_results[estimator.getRunId()]['trial'] = trial
                            for epoch in range(epochs):
                                train_epoch = ms.backend.train_for_one_epoch([estimator], ms.store, None, ms.feature_cols, ms.label_cols)
                                update_model_results(est_results, train_epoch)

                                val_epoch = ms.backend.train_for_one_epoch([estimator], ms.store, None, ms.feature_cols, ms.label_cols, is_train=False)
                                update_model_results(est_results, val_epoch)
                            i = i+1
        return est_results 

    def trials2estimators(self, trials, dataset):
        ests = []
        for trial in trials:
            self._prepare_model_IO(trial.hyperparameters, dataset=dataset)
            model = self.hypermodel.build(trial.hyperparameters)
            self.adapt(model, dataset)
            hm = self.hypermodel.hypermodel
            loss_dict = hm._get_loss()
            metric_dict = hm._get_metrics()
            loss = []
            metric = []
            for lname in loss_dict:
                loss.append(loss_dict[lname])
            for mname in metric_dict:
                metric.extend(metric_dict[mname])
            if len(loss) > 1:
                raise ValueError("Multiple losses detected")
            # if len(metric) > 1:
            #     raise ValueError("Multiple metrics detected")
            params = {
                'model': model,
                'optimizer': model.optimizer, # keras opt not str
                'loss': loss[0], # not sure
                'metrics': metric,
                'batch_size': hm.batch_size,
                'custom_objects': tf.keras.utils.get_custom_objects()
            }
            est = self.model_selection._estimator_gen_fn_wrapper(params)
            ests.append(est)
        return ests

    def trial_from_config_to_est(
        self,
        trial,
        dataset,
        **kwargs
    ):
        self._prepare_model_IO(trial.hyperparameters, dataset=dataset)
        model = self.hypermodel.build(trial.hyperparameters)
        self.adapt(model, dataset)
        hm = self.hypermodel.hypermodel
        loss_dict = hm._get_loss()
        metric_dict = hm._get_metrics()
        loss = []
        metric = []
        for lname in loss_dict:
            loss.append(loss_dict[lname])
        for mname in metric_dict:
            metric.extend(metric_dict[mname])
        if len(loss) > 1:
            raise ValueError("Multiple losses detected")
        # if len(metric) > 1:
        #     raise ValueError("Multiple metrics detected")
        lr = kwargs['learning_rate']
        batch_size = kwargs['batch_size']
        optimizer_name = kwargs['optimizer']

        if optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_name == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        elif optimizer_name == "adam_weight_decay":
            num_samples = kwargs['num_samples']
            epochs = kwargs['epochs']
            steps_per_epoch = int(num_samples / batch_size)
            num_train_steps = steps_per_epoch * epochs
            warmup_steps = int(
                epochs * num_samples * 0.1 / batch_size
            )

            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=lr,
                decay_steps=num_train_steps,
                end_learning_rate=0.0,
            )
            if warmup_steps:
                lr_schedule = keras_layers.WarmUp(
                    initial_learning_rate=lr,
                    decay_schedule_fn=lr_schedule,
                    warmup_steps=warmup_steps,
                )

            optimizer = keras_layers.AdamWeightDecay(
                learning_rate=lr_schedule,
                weight_decay_rate=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
                exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            )            

        params = {
            'model': model,
            'optimizer': optimizer, # keras opt not str
            'loss': loss[0], # not sure
            'metrics': metric,
            'batch_size': batch_size,
            'custom_objects': tf.keras.utils.get_custom_objects()
        }
        est = self.model_selection._estimator_gen_fn_wrapper(params)
        return est


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

    @staticmethod
    def adapt(model, dataset):
        """Adapt the preprocessing layers in the model."""
        # Currently, only support using the original dataset to adapt all the
        # preprocessing layers before the first non-preprocessing layer.
        # TODO: Use PreprocessingStage for preprocessing layers adapt.
        # TODO: Use Keras Tuner for preprocessing layers adapt.
        x = dataset.map(lambda x, y: x)

        def get_output_layers(tensor):
            output_layers = []
            tensor = nest.flatten(tensor)[0]
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                input_node = nest.flatten(layer.input)[0]
                if input_node is tensor:
                    if isinstance(layer, preprocessing.PreprocessingLayer):
                        output_layers.append(layer)
            return output_layers

        dq = collections.deque()

        for index, input_node in enumerate(nest.flatten(model.input)):
            in_x = x.map(lambda *args: nest.flatten(args)[index])
            for layer in get_output_layers(input_node):
                dq.append((layer, in_x))

        while len(dq):
            layer, in_x = dq.popleft()
            layer.adapt(in_x)
            out_x = in_x.map(layer)
            for next_layer in get_output_layers(layer.output):
                dq.append((next_layer, out_x))

        return model

    @property
    def objective(self):
        return self.oracle.objective

    @property
    def max_trials(self):
        return self.oracle.max_trials