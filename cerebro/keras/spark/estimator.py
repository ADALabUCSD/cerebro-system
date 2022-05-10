# Copyright 2020 Supun Nakandala, Yuhao Zhang, and Arun Kumar. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

from __future__ import absolute_import

import numbers
import time
import os
import json
import inspect
import numpy as np
import tensorflow as tf

from pyspark import keyword_only
from pyspark.ml.util import MLWritable, MLReadable
from pyspark.ml.param.shared import Param, Params
from pyspark.ml import Estimator as PySparkEstimator, Model as PySparkModel
from pyspark.ml.util import DefaultParamsWriter, DefaultParamsReader

from ...backend import codec
from ...backend import spark
from ...commons.util import patch_hugginface_layer_methods
from .util import TF_KERAS, TFKerasUtil
from .params import SparkEstimatorParams, SparkModelParams
from ..estimator import CerebroEstimator, CerebroModel

import threading

LOCK = threading.Lock()
MODEL_ID = -1


def next_model_id():
    global LOCK, MODEL_ID
    with LOCK:
        MODEL_ID += 1
        return MODEL_ID


class KerasEstimatorParamsWriter(DefaultParamsWriter):
    @staticmethod
    def saveMetadata(instance, path, sc, extraMetadata=None, paramMap=None,
                     param_serializer_fn=None):
        metadata_path = os.path.join(path, "metadata")
        metadata_json = KerasEstimatorParamsWriter. \
            _get_metadata_to_save(instance,
                                  sc,
                                  extraMetadata,
                                  paramMap,
                                  param_serializer_fn)
        sc.parallelize([metadata_json], 1).saveAsTextFile(metadata_path)

    @staticmethod
    def _get_metadata_to_save(instance, sc, extra_metadata=None, param_map=None,
                              param_serializer_fn=None):
        uid = instance.uid
        cls = instance.__module__ + '.' + instance.__class__.__name__

        # User-supplied param values
        params = instance._paramMap
        json_params = {}
        if param_map is not None:
            json_params = param_map
        else:
            for p, param_val in params.items():
                # If param is not json serializable, convert it into serializable object
                json_params[p.name] = param_serializer_fn(p.name, param_val)

        # Default param values
        json_default_params = {}
        for p, param_val in instance._defaultParamMap.items():
            json_default_params[p.name] = param_serializer_fn(p.name,
                                                              param_val)

        basic_metadata = {"class": cls, "timestamp": int(round(time.time() * 1000)),
                          "sparkVersion": sc.version, "uid": uid, "paramMap": json_params,
                          "defaultParamMap": json_default_params}
        if extra_metadata is not None:
            basic_metadata.update(extra_metadata)
        return json.dumps(basic_metadata, separators=[',', ':'])

    def saveImpl(self, path):
        keras_utils = self.instance._get_keras_utils()
        # Write the parameters
        KerasEstimatorParamsWriter.saveMetadata(self.instance, path, self.sc,
                                                param_serializer_fn=keras_utils.serialize_param_value)


class KerasEstimatorParamsReader(DefaultParamsReader):
    def load(self, path):
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        metadata['paramMap'] = self._deserialize_dict(metadata['paramMap'])
        metadata['defaultParamMap'] = self._deserialize_dict(metadata['defaultParamMap'])

        py_type = DefaultParamsReader._DefaultParamsReader__get_class(metadata['class'])
        instance = py_type()
        instance._resetUid(metadata['uid'])
        DefaultParamsReader.getAndSetParams(instance, metadata)
        return instance

    def _deserialize_dict(self, dict):
        def _param_deserializer_fn(name, param_val, keras_utils, custom_objects):
            if param_val is None:
                return param_val

            if name == SparkEstimatorParams.model.name:
                def load_model_fn(x):
                    with keras_utils.keras().utils.custom_object_scope(custom_objects):
                        return keras_utils.keras().models.load_model(x, compile=True)

                return keras_utils.deserialize_model(param_val,
                                                     load_model_fn=load_model_fn)
            elif name == SparkEstimator.optimizer.name:
                opt_base64_encoded = codec.loads_base64(param_val)
                return keras_utils.deserialize_optimizer(opt_base64_encoded)
            else:
                return codec.loads_base64(param_val)

        # In order to deserialize the model, we need to deserialize the custom_objects param
        # first.
        keras_utils = None
        if SparkEstimator._keras_pkg_type.name in dict:
            keras_pkg_type = _param_deserializer_fn(SparkEstimator._keras_pkg_type.name,
                                                    dict[SparkEstimator._keras_pkg_type.name],
                                                    None, None)
            if keras_pkg_type == TF_KERAS:
                keras_utils = TFKerasUtil
            else:
                raise ValueError("invalid keras type")

        custom_objects = {}
        if SparkEstimator.custom_objects.name in dict:
            custom_objects = _param_deserializer_fn(SparkEstimator.custom_objects.name,
                                                    dict[SparkEstimator.custom_objects.name],
                                                    None, None)

        for key, val in dict.items():
            dict[key] = _param_deserializer_fn(key, val, keras_utils, custom_objects)
        return dict


class SparkEstimatorParamsWritable(MLWritable):
    def write(self):
        return KerasEstimatorParamsWriter(self)


class SparkEstimatorParamsReadable(MLReadable):
    @classmethod
    def read(cls):
        """Returns a KerasEstimatorParamsReader instance for this class."""
        return KerasEstimatorParamsReader(cls)


class SparkEstimator(PySparkEstimator, SparkEstimatorParams, SparkEstimatorParamsReadable, SparkEstimatorParamsWritable,
                     CerebroEstimator):
    """Cerebro Spark Estimator for fitting Keras model to a DataFrame.

    Supports ``tf.keras >= 2.2``.

    Args:
        model: Keras model to train.
        custom_objects: Optional dictionary mapping names (strings) to custom classes or functions to be considered
                        during serialization/deserialization.
        optimizer: Keras optimizer.
        loss: Keras loss or list of losses.
        batch_size: Number of rows from the DataFrame per batch.
        loss_weights: (Optional) List of float weight values to assign each loss.
        metrics: (Optional) List of Keras metrics to record.
        callbacks: (Optional) List of Keras callbacks.
        transformation_fn: (Optional) Function that takes a TensorFlow Dataset as its parameter
                       and returns a modified Dataset that is then fed into the
                       train or validation step. This transformation is applied before batching.
    """
    
    custom_objects = Param(Params._dummy(), 'custom_objects', 'custom objects')
    _keras_pkg_type = Param(Params._dummy(), '_keras_pkg_type', 'keras package type')

    @keyword_only
    def __init__(self,
                 model=None,
                 custom_objects=None,
                 optimizer=None,
                 loss=None,
                 batch_size=None,
                 loss_weights=None,
                 metrics=None,
                 callbacks=None,
                 transformation_fn=None
                 ):

        super(SparkEstimator, self).__init__()

        self._setDefault(optimizer=None,
                         custom_objects={},
                         _keras_pkg_type=None)

        run_id = 'model_' + str(next_model_id()) + '_' + str(int(time.time()))
        self.setRunId(run_id)

        # Initializing the epochs to 0
        self.setEpochs(0)

        kwargs = self._input_kwargs
        self.setParams(**kwargs)


    def _fit(self):
        raise NotImplementedError

    def _get_keras_utils(self):
        # This function determines the keras package type of the Estimator based on the passed
        # optimizer and model and updates _keras_pkg_type parameter.

        model_type = None
        model = self.getModel()
        if model:
            if isinstance(model, tf.keras.Model):
                model_type = TF_KERAS
            else:
                raise ValueError(
                    "model has to be an instance of tensorflow.keras.Model or keras.Model")

        optimizer_type = None
        optimizer = self.getOptimizer()
        if optimizer:
            if isinstance(optimizer, str):
                optimizer_type = None
            elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
                optimizer_type = TF_KERAS
            else:
                raise ValueError("invalid optimizer type")

        types = set([model_type, optimizer_type])
        types.discard(None)

        if len(types) > 1:
            raise ValueError('mixed keras and tf.keras values for optimizers and model')
        elif len(types) == 1:
            pkg_type = types.pop()
            super(SparkEstimator, self)._set(_keras_pkg_type=pkg_type)

            if pkg_type == TF_KERAS:
                return TFKerasUtil
            else:
                raise ValueError("invalid keras type")

    def setCustomObjects(self, value):
        return self._set(custom_objects=value)

    def getCustomObjects(self):
        return self.getOrDefault(self.custom_objects)

    def _check_metadata_compatibility(self, metadata):
        input_shapes, output_shapes = self.get_model_shapes()
        spark.util.check_shape_compatibility(metadata,
                                             self.getFeatureCols(),
                                             self.getLabelCols(),
                                             input_shapes=input_shapes,
                                             output_shapes=output_shapes)

    def get_model_shapes(self):
        model = self.getModel()
        input_shapes = [[dim if dim else -1 for dim in input.shape.as_list()]
                        for input in model.inputs]
        output_shapes = [[dim if dim else -1 for dim in output.shape.as_list()]
                         for output in model.outputs]
        return input_shapes, output_shapes


    def _load_model_from_checkpoint(self, run_id):
        store = self.getStore()
        last_ckpt_path = store.get_checkpoint_path(run_id)

        model_bytes = store.read(last_ckpt_path)
        return codec.dumps_base64(model_bytes)

    def _compile_model(self, keras_utils):
        # Compile the model with all the parameters
        model = self.getModel()

        loss = self.getLoss()
        loss_weights = self.getLossWeights()

        if not loss:
            raise ValueError('Loss parameter is required for the model to compile')

        optimizer = self.getOptimizer()
        if not optimizer:
            optimizer = model.optimizer

        if not optimizer:
            raise ValueError('Optimizer must be provided either as a parameter or as part of a '
                             'compiled model')

        metrics = self.getMetrics()
        optimizer_weight_values = optimizer.get_weights()

        model.compile(optimizer=optimizer,
                      loss=loss,
                      loss_weights=loss_weights,
                      metrics=metrics)

        if optimizer_weight_values:
            model.optimizer.set_weights(optimizer_weight_values)

        return model
        # return keras_utils.serialize_model(model)

    def create_model(self, history, run_id, metadata):
        keras_utils = self._get_keras_utils()
        keras_module = keras_utils.keras()
        floatx = keras_module.backend.floatx()
        custom_objects = self.getCustomObjects()
        serialized_model = self._load_model_from_checkpoint(run_id)

        def load_model_fn(x):
            with keras_module.utils.custom_object_scope(custom_objects):
                return keras_module.models.load_model(x)

        model = keras_utils.deserialize_model(serialized_model, load_model_fn=load_model_fn)
        return self.get_model_class()(**self._get_model_kwargs(model, history, run_id, metadata, floatx))

    def get_model_class(self):
        return SparkModel

    def _get_model_kwargs(self, model, history, run_id, metadata, floatx):
        return dict(history=history,
                    model=model,
                    feature_columns=self.getFeatureCols(),
                    label_columns=self.getLabelCols(),
                    custom_objects=self.getCustomObjects(),
                    run_id=run_id,
                    _metadata=metadata,
                    _floatx=floatx)

    def _has_checkpoint(self, run_id):
        store = self.getStore()
        last_ckpt_path = store.get_checkpoint_path(run_id)
        return last_ckpt_path is not None and store.exists(last_ckpt_path)


class SparkModel(PySparkModel, SparkModelParams, SparkEstimatorParamsReadable, SparkEstimatorParamsWritable, CerebroModel):
    """Spark Transformer wrapping a Keras model, used for making predictions on a DataFrame.

    Retrieve the underlying Keras model by calling ``keras_model.getModel()``.

    Args:
        history: List of metrics, one entry per epoch during training.
        model: Trained Keras model.
        feature_columns: List of feature column names.
        label_columns: List of label column names.
        custom_objects: Keras custom objects.
        run_id: ID of the run used to train the model.
    """

    custom_objects = Param(Params._dummy(), 'custom_objects', 'custom objects')

    # Setting _keras_pkg_type parameter helps us determine the type of keras package during
    # deserializing the transformer
    _keras_pkg_type = Param(Params._dummy(), '_keras_pkg_type', 'keras package type')

    _floatx = Param(Params._dummy(), '_floatx', 'keras default float type')

    @keyword_only
    def __init__(self,
                 history=None,
                 model=None,
                 feature_columns=None,
                 label_columns=None,
                 custom_objects=None,
                 run_id=None,
                 _metadata=None,
                 _floatx=None):

        super(SparkModel, self).__init__()

        if label_columns:
            self.setOutputCols([col + '__output' for col in label_columns])

        self._setDefault(custom_objects={})

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def setCustomObjects(self, value):
        return self._set(custom_objects=value)

    def getCustomObjects(self):
        return self.getOrDefault(self.custom_objects)

    def _get_keras_utils(self):
        # infer keras package from model
        model = self.getModel()
        if model:
            if isinstance(model, tf.keras.Model):
                pkg_type = TF_KERAS
            else:
                raise ValueError(
                    "model has to be an instance of tensorflow.keras.Model")

            super(SparkModel, self)._set(_keras_pkg_type=pkg_type)

            if pkg_type == TF_KERAS:
                return TFKerasUtil
            else:
                raise ValueError("invalid keras type")

        raise ValueError("model is not set")

    def _get_floatx(self):
        return self.getOrDefault(self._floatx)


    def _transform(self):
        """ Only required by PySparkModel, not used at all
        """
        raise NotImplementedError

    def transform(self, df):
        """ Transforms the given DataFrame using the trained ML model

        :param df: Input DataFrame
        :return: Transformed DataFrame
        """
        keras_utils = self._get_keras_utils()
        floatx = self._get_floatx()
        serialized_model = keras_utils.serialize_model(self.getModel())

        label_cols = self.getLabelColumns()
        output_cols = self.getOutputCols()
        feature_cols = self.getFeatureColumns()
        custom_objects = self.getCustomObjects()
        metadata = self._get_metadata()

        pin_cpu = spark.backend._pin_cpu_fn()

        def predict(rows):
            import tensorflow as tf
            from pyspark import Row
            from pyspark.ml.linalg import DenseVector, SparseVector

            k = keras_utils.keras()
            k.backend.set_floatx(floatx)
            # Do not use GPUs for prediction, use single CPU core per task.
            pin_cpu(tf, k)

            def load_model_fn(x):
                with k.utils.custom_object_scope(custom_objects):
                    return k.models.load_model(x)

            model = keras_utils.deserialize_model(serialized_model,
                                                  load_model_fn=load_model_fn)

            input_shapes = [[dim if dim else -1 for dim in input.shape.as_list()]
                            for input in model.inputs]

            def to_array(item):
                if type(item) in [DenseVector or SparseVector]:
                    return np.array(item.toArray())
                else:
                    return np.array(item)

            def to_numpy(item):
                # Some versions of TensorFlow will return an EagerTensor
                return item.numpy() if hasattr(item, 'numpy') else item

            # Perform predictions.
            for row in rows:
                fields = row.asDict().copy()
                preds = model.predict_on_batch(
                    [to_array(row[feature_cols[i]]).reshape(input_shapes[i])
                     for i in range(len(feature_cols))])

                preds = [to_numpy(item) for item in preds]

                for label_col, output_col, pred, in zip(label_cols, output_cols, preds):
                    meta = metadata[label_col]
                    col_type = meta['spark_data_type']
                    # dtype for DenseVector and SparseVector is always np.float64
                    if col_type == DenseVector:
                        shape = np.prod(pred.shape)
                        flattened_pred = pred.reshape(shape, )
                        field = DenseVector(flattened_pred)
                    elif col_type == SparseVector:
                        shape = meta['shape']
                        flattened_pred = pred.reshape(shape, )
                        nonzero_indices = flattened_pred.nonzero()[0]
                        field = SparseVector(shape, nonzero_indices,
                                             flattened_pred[nonzero_indices])
                    else:
                        # If the column is scalar type, int, float, etc.
                        value = pred[0]
                        python_type = spark.util.spark_scalar_to_python_type(col_type)
                        if issubclass(python_type, numbers.Integral):
                            field = round(value.item())
                        else:
                            field = value.item()
                        # field = python_type(value)

                    fields[output_col] = field

                yield Row(**fields)

        return df.rdd.mapPartitions(predict).toDF()

    def keras(self):
        """ Returns the trained model in Keras format.

            :return: TensorFlow Keras Model
        """
        if self.model is not None:
            return self.getModel()
        else:
            raise Exception('Keras model is not set!')

    # copied from https://github.com/apache/spark/tree/master/python/pyspark/ml/param/shared.py
    # has been removed from pyspark.ml.param.HasOutputCol in pyspark 3.0.0
    # added here to keep ModelParams API consistent between pyspark 2 and 3
    # https://github.com/apache/spark/commit/b19fd487dfe307542d65391fd7b8410fa4992698#diff-3d1fb305acc7bab18e5d91f2b69018c7
    # https://github.com/apache/spark/pull/26232
    # https://issues.apache.org/jira/browse/SPARK-29093
    def setOutputCols(self, value):
        """
        Sets the value of :py:attr:`outputCols`.
        """
        return self._set(outputCols=value)
