
from .tuners.randsearch import RandomSearch
from .tuners.gridsearch import GridSearch
# from .tuners.hyperband import Hyperband
from .sparktuner import SparkTuner
from ..keras.spark.estimator import SparkEstimator

from pathlib import Path
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import datetime

from autokeras import blocks, ImageBlock
from autokeras import graph as graph_module
from autokeras import pipeline
from autokeras import tuners
from autokeras import auto_model
from autokeras.engine import head as head_module
from autokeras.engine import node as node_module
from autokeras.engine import tuner
from autokeras.nodes import Input
from autokeras.utils import data_utils
from autokeras.utils import utils
from keras_tuner import HyperParameters

from ..tune.base import ModelSelection

NAS_TUNERS = {
    # "hyperband": Hyperband,
    "gridsearch": GridSearch,
    "randomsearch": RandomSearch,
}

def get_tuner_class(tuner):
    if isinstance(tuner, str) and tuner in NAS_TUNERS:
        return NAS_TUNERS.get(tuner)
    else:
        raise ValueError(
            'Expected the tuner argument to be one of "greedy", '
            '"random", "hyperband", or "bayesian", '
            "but got {tuner}".format(tuner=tuner)
        )

class HyperHyperModel(object):
    
    def __init__(
        self,
        inputs: Union[Input, List[Input]],
        outputs: Union[head_module.Head, node_module.Node, list],
        seed: Optional[int] = None,
        **kwargs
    ):
        self.inputs = nest.flatten(inputs)
        self.outputs = nest.flatten(outputs)
        self.seed = seed
        if seed:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        # TODO: Support passing a tuner instance.
        # Initialize the hyper_graph.
        self.graph = self._build_graph()
        self._heads = [output_node.in_blocks[0] for output_node in self.outputs]

    def resource_bind(self,
        backend,
        store,
        validation=0.2, 
        evaluation_metric='loss', 
        label_columns='labels', 
        feature_columns='features', 
        verbose=1):

        """
        Default estimator_generation_function used to initialize model_selection class, all parameters are extracted from params
        """
        def est_gen_fun(params):
            model = params['model']
            opt = params['optimizer']
            loss = params['loss']
            metrics = params['metrics']
            bs = params['batch_size']
            cuz_obj_dict = params['custom_objects']
            estimator = SparkEstimator(
                model=model,
                optimizer=opt,
                loss=loss,
                metrics=metrics,
                batch_size=bs,
                custom_objects=cuz_obj_dict
            )
            return estimator

        """
        We temporarily keep the entire ModelSelection class here for compatibility and potential usage of its methods
        It encapsulates backend engine which shall be used in the training process
        """
        self.model_selection = ModelSelection(
            backend,
            store,
            validation,
            est_gen_fun,
            evaluation_metric,
            label_columns,
            feature_columns,
            verbose
        )

    def tuner_bind(self, 
        parallelism = None,
        tuner: str = "gridsearch",
        project_name: str = "test",
        max_trials: int = 100,
        directory: Union[str, Path, None] = None,
        objective: str = "val_loss",
        overwrite: bool = False,
        max_model_size: Optional[int] = None,
        hyperparameters: Optional[HyperParameters] = None,
        **kwargs):
        if isinstance(tuner, str):
            tuner = get_tuner_class(tuner)
        else:
            # return exception
            pass
        if parallelism is None:
            parallelism = self.model_selection.backend._num_workers()
        self.tuner = tuner(
            hypermodel=self.graph,
            parallelism = parallelism,
            hyperparameters=hyperparameters,
            model_selection=self.model_selection,
            overwrite=overwrite,
            objective=objective,
            max_trials=max_trials,
            directory=directory,
            seed=self.seed,
            project_name=project_name,
            max_model_size=max_model_size,
            **kwargs
        )

    @property
    def objective(self):
        return self.tuner.objective

    @property
    def max_trials(self):
        return self.tuner.max_trials

    @property
    def directory(self):
        return self.tuner.directory

    @property
    def project_name(self):
        return self.tuner.project_name

    def _assemble(self):
        """Assemble the Blocks based on the input output nodes."""
        inputs = nest.flatten(self.inputs)
        outputs = nest.flatten(self.outputs)

        middle_nodes = [input_node.get_block()(input_node) for input_node in inputs]

        # Merge the middle nodes.
        if len(middle_nodes) > 1:
            output_node = blocks.Merge()(middle_nodes)
        else:
            output_node = middle_nodes[0]

        outputs = nest.flatten(
            [output_blocks(output_node) for output_blocks in outputs]
        )
        return graph_module.Graph(inputs=inputs, outputs=outputs)

    def _build_graph(self):
        # Using functional API.
        if all([isinstance(output, node_module.Node) for output in self.outputs]):
            graph = graph_module.Graph(inputs=self.inputs, outputs=self.outputs)
        # Using input/output API.
        elif all([isinstance(output, head_module.Head) for output in self.outputs]):
            graph = self._assemble()
            self.outputs = graph.outputs

        return graph

    def fit(
        self,
        df,
        batch_size=32,
        epochs=None,
        callbacks=None,
        verbose=1,
        input_shape = None,
        **kwargs
    ):

        """
        Setup cluster before calling the tuner
        """
        ms = self.model_selection
        backend = ms.backend

        if ms.verbose >= 1: print(
            'CEREBRO => Time: {}, Preparing Data'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        _, _, metadata, _ = backend.prepare_data(ms.store, df, ms.validation, label_columns=ms.label_cols, feature_columns=ms.feature_cols)

        if ms.verbose >= 1: print(
            'CEREBRO => Time: {}, Initializing Workers'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        # initialize backend and data loaders
        backend.initialize_workers()

        if ms.verbose >= 1: print(
            'CEREBRO => Time: {}, Initializing Data Loaders'.format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        backend.initialize_data_loaders(ms.store, None, ms.feature_cols + ms.label_cols)
        if input_shape and type(input_shape) is int or type(input_shape) is tuple:
            x = np.array(df.select(ms.feature_cols).head(100))
            y = np.array(df.select(ms.label_cols).head(100))
            x = [x[:,i] for i in range(x.shape[1])]
            x = [r.reshape((-1, *input_shape)) for r in x]
            y = np.squeeze(y,1)
        else:
            x = np.array(df.select(ms.feature_cols).head(100))
            y = np.array(df.select(ms.label_cols).head(100))
            x = [x[:,i,...,np.newaxis] for i in range(x.shape[1])]
            y = np.squeeze(y,1)
        if len(y.shape) > 2:
            raise ValueError(
                "We do not support multiple labels. Expect the target data for {name} to have shape "
                "(batch_size, num_classes), "
                "but got {shape}.".format(name=self.name, shape=self.shape)
            )
        dataset, validation_data = self._convert_to_dataset(
            x=x, y=y, validation_data=None, batch_size=batch_size
        )

        """
        Analyze data analyse input and output data and config model inputs and heads
        """
        self._analyze_data(dataset)

        """
        Build preprocessing pipeline with tunable parameters

        Since the model is trained from workers which reads data from pre-distributed permanent storage, we will not consider tuning preprocessing currently.
        """
        # self._build_hyper_pipeline(dataset)
        self.tuner.hyper_pipeline = None
        self.tuner.hypermodel.hyper_pipeline = None
        return self.tuner.search(
            x=dataset,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_split=ms.validation,
            verbose=verbose,
            dataset_idx=None,
            metadata=metadata,
            **kwargs
        )

        # return history

    def _adapt(self, dataset, hms, batch_size):
        if isinstance(dataset, tf.data.Dataset):
            sources = data_utils.unzip_dataset(dataset)
        else:
            sources = nest.flatten(dataset)
        adapted = []
        for source, hm in zip(sources, hms):
            source = hm.get_adapter().adapt(source, batch_size)
            adapted.append(source)
        if len(adapted) == 1:
            return adapted[0]
        return tf.data.Dataset.zip(tuple(adapted))

    def _check_data_format(self, dataset, validation=False, predict=False):
        """Check if the dataset has the same number of IOs with the model."""
        if validation:
            in_val = " in validation_data"
            if isinstance(dataset, tf.data.Dataset):
                x = dataset
                y = None
            else:
                x, y = dataset
        else:
            in_val = ""
            x, y = dataset

        if isinstance(x, tf.data.Dataset) and y is not None:
            raise ValueError(
                "Expected y to be None when x is "
                "tf.data.Dataset{in_val}.".format(in_val=in_val)
            )

        if isinstance(x, tf.data.Dataset):
            if not predict:
                x_shapes, y_shapes = data_utils.dataset_shape(x)
                x_shapes = nest.flatten(x_shapes)
                y_shapes = nest.flatten(y_shapes)
            else:
                x_shapes = nest.flatten(data_utils.dataset_shape(x))
        else:
            x_shapes = [a.shape for a in nest.flatten(x)]
            if not predict:
                y_shapes = [a.shape for a in nest.flatten(y)]

        if len(x_shapes) != len(self.inputs):
            raise ValueError(
                "Expected x{in_val} to have {input_num} arrays, "
                "but got {data_num}".format(
                    in_val=in_val, input_num=len(self.inputs), data_num=len(x_shapes)
                )
            )
        if not predict and len(y_shapes) != len(self.outputs):
            raise ValueError(
                "Expected y{in_val} to have {output_num} arrays, "
                "but got {data_num}".format(
                    in_val=in_val,
                    output_num=len(self.outputs),
                    data_num=len(y_shapes),
                )
            )

    def _analyze_data(self, dataset):
        input_analysers = [node.get_analyser() for node in self.inputs]
        output_analysers = [head.get_analyser() for head in self._heads]
        analysers = input_analysers + output_analysers
        for x, y in dataset:
            x = nest.flatten(x)
            y = nest.flatten(y)
            for item, analyser in zip(x + y, analysers):
                analyser.update(item)

        for analyser in analysers:
            analyser.finalize()

        for hm, analyser in zip(self.inputs + self._heads, analysers):
            hm.config_from_analyser(analyser)

    def _build_hyper_pipeline(self, dataset):
        self.tuner.hyper_pipeline = pipeline.HyperPipeline(
            inputs=[node.get_hyper_preprocessors() for node in self.inputs],
            outputs=[head.get_hyper_preprocessors() for head in self._heads],
        )
        self.tuner.hypermodel.hyper_pipeline = self.tuner.hyper_pipeline

    def _convert_to_dataset(self, x, y, validation_data, batch_size):
        """Convert the data to tf.data.Dataset."""
        # TODO: Handle other types of input, zip dataset, tensor, dict.

        # Convert training data.
        self._check_data_format((x, y))
        if isinstance(x, tf.data.Dataset):
            dataset = x
            x = dataset.map(lambda x, y: x)
            y = dataset.map(lambda x, y: y)
        x = self._adapt(x, self.inputs, batch_size)
        y = self._adapt(y, self._heads, batch_size)
        dataset = tf.data.Dataset.zip((x, y))
        # Convert validation data
        if validation_data:
            self._check_data_format(validation_data, validation=True)
            if isinstance(validation_data, tf.data.Dataset):
                x = validation_data.map(lambda x, y: x)
                y = validation_data.map(lambda x, y: y)
            else:
                x, y = validation_data
            x = self._adapt(x, self.inputs, batch_size)
            y = self._adapt(y, self._heads, batch_size)
            validation_data = tf.data.Dataset.zip((x, y))

        return dataset, validation_data

    def _has_y(self, dataset):
        """Remove y from the tf.data.Dataset if exists."""
        shapes = data_utils.dataset_shape(dataset)
        # Only one or less element in the first level.
        if len(shapes) <= 1:
            return False
        # The first level has more than 1 element.
        # The nest has 2 levels.
        for shape in shapes:
            if isinstance(shape, tuple):
                return True
        # The nest has one level.
        # It matches the single IO case.
        return len(shapes) == 2 and len(self.inputs) == 1 and len(self.outputs) == 1

    def predict(self, x, batch_size=32, verbose=1, **kwargs):
        """Predict the output for a given testing data.

        # Arguments
            x: Any allowed types according to the input node. Testing data.
            batch_size: Number of samples per batch.
                If unspecified, batch_size will default to 32.
            verbose: Verbosity mode. 0 = silent, 1 = progress bar.
                Controls the verbosity of
                [keras.Model.predict](https://tensorflow.org/api_docs/python/tf/keras/Model#predict)
            **kwargs: Any arguments supported by keras.Model.predict.

        # Returns
            A list of numpy.ndarray objects or a single numpy.ndarray.
            The predicted results.
        """
        if isinstance(x, tf.data.Dataset) and self._has_y(x):
            x = x.map(lambda x, y: x)
        self._check_data_format((x, None), predict=True)
        dataset = self._adapt(x, self.inputs, batch_size)
        pipeline = self.tuner.get_best_pipeline()
        model = self.tuner.get_best_model()
        dataset = pipeline.transform_x(dataset)
        dataset = tf.data.Dataset.zip((dataset, dataset))
        y = model.predict(dataset, **kwargs)
        y = utils.predict_with_adaptive_batch_size(
            model=model, batch_size=batch_size, x=dataset, verbose=verbose, **kwargs
        )
        return pipeline.postprocess(y)

    def evaluate(self, x, y=None, batch_size=32, verbose=1, **kwargs):
        """Evaluate the best model for the given data.

        # Arguments
            x: Any allowed types according to the input node. Testing data.
            y: Any allowed types according to the head. Testing targets.
                Defaults to None.
            batch_size: Number of samples per batch.
                If unspecified, batch_size will default to 32.
            verbose: Verbosity mode. 0 = silent, 1 = progress bar.
                Controls the verbosity of
                [keras.Model.evaluate](http://tensorflow.org/api_docs/python/tf/keras/Model#evaluate)
            **kwargs: Any arguments supported by keras.Model.evaluate.

        # Returns
            Scalar test loss (if the model has a single output and no metrics) or
            list of scalars (if the model has multiple outputs and/or metrics).
            The attribute model.metrics_names will give you the display labels for
            the scalar outputs.
        """
        self._check_data_format((x, y))
        if isinstance(x, tf.data.Dataset):
            dataset = x
            x = dataset.map(lambda x, y: x)
            y = dataset.map(lambda x, y: y)
        x = self._adapt(x, self.inputs, batch_size)
        y = self._adapt(y, self._heads, batch_size)
        dataset = tf.data.Dataset.zip((x, y))
        pipeline = self.tuner.get_best_pipeline()
        dataset = pipeline.transform(dataset)
        model = self.tuner.get_best_model()
        return utils.evaluate_with_adaptive_batch_size(
            model=model, batch_size=batch_size, x=dataset, verbose=verbose, **kwargs
        )

    def export_model(self):
        """Export the best Keras Model.

        # Returns
            tf.keras.Model instance. The best model found during the search, loaded
            with trained weights.
        """
        return self.tuner.get_best_model()

    def test_tuner_space(
        self,
        df,
        batch_size=32,
        epochs=100,
        **kwargs
    ):
        ms = self.model_selection
        x = np.array(df.select(ms.feature_cols).head(100))
        y = np.array(df.select(ms.label_cols).head(100))
        x = [x[:,i,...,np.newaxis] for i in range(x.shape[1])]
        y = np.squeeze(y,1)
        dataset, validation_data = self._convert_to_dataset(
            x=x, y=y, validation_data=None, batch_size=batch_size
        )
        self._analyze_data(dataset)
        self.tuner.space_initialize_test(
            x=dataset,
            validation_split=0.2,
            epochs=epochs,
            **kwargs
        )

    def sys_setup(self, df):
        ms = self.model_selection
        backend = ms.backend

        if ms.verbose >= 1: print(
            'CEREBRO => Time: {}, Preparing Data'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        train_rows, val_rows, metadata, avg_row_size = backend.prepare_data(ms.store, df, ms.validation, label_columns=ms.label_cols, feature_columns=ms.feature_cols)

        if ms.verbose >= 1: print(
            'CEREBRO => Time: {}, Initializing Workers'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        # initialize backend and data loaders
        backend.initialize_workers()

        if ms.verbose >= 1: print(
            'CEREBRO => Time: {}, Initializing Data Loaders'.format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        backend.initialize_data_loaders(ms.store, None, ms.feature_cols + ms.label_cols)
        return train_rows, val_rows, metadata, avg_row_size