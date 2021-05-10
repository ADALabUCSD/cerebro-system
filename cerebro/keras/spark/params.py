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

from pyspark import keyword_only
from pyspark.ml.param.shared import HasOutputCols, Param, Params, TypeConverters

from ..params import CerebroEstimatorParams, CerebroModelParams


def _check_validation(validation):
    if validation:
        if isinstance(validation, float):
            if validation < 0 or validation >= 1:
                raise ValueError('Validation split {} must be in the range: [0, 1)'
                                 .format(validation))
        else:
            raise ValueError('Param validation must be of type "float", found: {}'
                             .format(type(validation)))


class SparkEstimatorParams(Params, CerebroEstimatorParams):
    optimizer = Param(Params._dummy(), 'optimizer', 'optimizer')
    model = Param(Params._dummy(), 'model', 'model')
    hyper_params = Param(Params._dummy(), 'hyper_params', 'Hyperparameters for this estimator model.')
    store = Param(Params._dummy(), 'store', 'store')
    metrics = Param(Params._dummy(), 'metrics', 'metrics')
    loss = Param(Params._dummy(), 'loss', 'loss')

    """loss_weights: Optional list of float weight values to assign each loss."""
    loss_weights = Param(Params._dummy(), 'loss_weights', 'loss weights',
                         typeConverter=TypeConverters.toListFloat)
    feature_cols = Param(Params._dummy(), "feature_cols", "feature column names",
                         typeConverter=TypeConverters.toListString)
    label_cols = Param(Params._dummy(), 'label_cols', 'label column names',
                       typeConverter=TypeConverters.toListString)
    validation = Param(Params._dummy(), 'validation',
                       'one of: float validation split [0, 1), or string validation column name',
                       typeConverter=TypeConverters.toString)
    callbacks = Param(Params._dummy(), 'callbacks', 'callbacks')
    batch_size = Param(Params._dummy(), 'batch_size', 'batch size',
                       typeConverter=TypeConverters.toInt)
    epochs = Param(Params._dummy(), 'epochs', 'epochs', typeConverter=TypeConverters.toInt)

    verbose = Param(Params._dummy(), 'verbose', 'verbose flag (0=silent, 1=enabled, other values used by frameworks)',
                    typeConverter=TypeConverters.toInt)

    run_id = Param(Params._dummy(), 'run_id', 'unique ID for this run', typeConverter=TypeConverters.toString)
    
    run_name = Param(Params._dummy(), 'run_name',
                   'unique name for this run, if checkpoint already exists for the run name, '
                   'then training will resume from last checkpoint in the store',
                   typeConverter=TypeConverters.toString)

    transformation_fn = Param(Params._dummy(), 'transformation_fn', '(Optional) Function that takes a TensorFlow Dataset as its parameter'
                                                                    ' and returns a modified Dataset that is then fed into the'
                                                                    ' train or validation step. This transformation is'
                                                                    ' applied before batching.')

    def __init__(self):
        super(SparkEstimatorParams, self).__init__()

        self._setDefault(
            store=None,
            model=None,
            optimizer=None,
            loss=None,
            loss_weights=None,
            metrics=[],
            feature_cols=None,
            label_cols=None,
            validation=None,
            batch_size=32,
            epochs=0,
            verbose=1,
            callbacks=[],
            run_id=None,
            transformation_fn=None
        )

    def _check_params(self, metadata):
        model = self.getModel()
        if not model:
            raise ValueError('Model parameter is required')

        _check_validation(self.getValidation())

        feature_columns = self.getFeatureCols()
        missing_features = [col for col in feature_columns if col not in metadata]
        if missing_features:
            raise ValueError('Feature columns {} not found in training DataFrame metadata'
                             .format(missing_features))

        label_columns = self.getLabelCols()
        missing_labels = [col for col in label_columns if col not in metadata]
        if missing_labels:
            raise ValueError('Label columns {} not found in training DataFrame metadata'
                             .format(missing_labels))

    @keyword_only
    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def setModel(self, value):
        return self._set(model=value)

    def getModel(self):
        return self.getOrDefault(self.model)

    def setHyperParams(self, value):
        return self._set(hyper_params=value)

    def getHyperParams(self):
        return self.getOrDefault(self.hyper_params)

    def setStore(self, value):
        return self._set(store=value)

    def getStore(self):
        return self.getOrDefault(self.store)

    def setLoss(self, value):
        return self._set(loss=value)

    def getLoss(self):
        return self.getOrDefault(self.loss)

    def setLossWeights(self, value):
        return self._set(loss_weights=value)

    def getLossWeights(self):
        return self.getOrDefault(self.loss_weights)

    def setMetrics(self, value):
        return self._set(metrics=value)

    def getMetrics(self):
        return self.getOrDefault(self.metrics)

    def setFeatureCols(self, value):
        return self._set(feature_cols=value)

    def getFeatureCols(self):
        return self.getOrDefault(self.feature_cols)

    def setLabelCols(self, value):
        return self._set(label_cols=value)

    def getLabelCols(self):
        return self.getOrDefault(self.label_cols)

    def setValidation(self, value):
        return self._set(validation=value)

    def getValidation(self):
        return self.getOrDefault(self.validation)

    def setCallbacks(self, value):
        return self._set(callbacks=value)

    def getCallbacks(self):
        return self.getOrDefault(self.callbacks)

    def setBatchSize(self, value):
        return self._set(batch_size=value)

    def getBatchSize(self):
        return self.getOrDefault(self.batch_size)

    def setEpochs(self, value):
        return self._set(epochs=value)

    def getEpochs(self):
        return self.getOrDefault(self.epochs)

    def setVerbose(self, value):
        return self._set(verbose=value)

    def getVerbose(self):
        return self.getOrDefault(self.verbose)

    def setOptimizer(self, value):
        return self._set(optimizer=value)

    def getOptimizer(self):
        return self.getOrDefault(self.optimizer)

    def setRunId(self, value):
        return self._set(run_id=value)

    def getRunId(self):
        return self.getOrDefault(self.run_id)

    def setRunName(self, value):
        return self._set(run_name=value)

    def getRunName(self):
        return self.getOrDefault(self.run_name)

    def setTransformationFn(self, value):
        return self._set(transformation_fn=value)

    def getTransformationFn(self):
        return self.getOrDefault(self.transformation_fn)


class SparkModelParams(HasOutputCols, CerebroModelParams):
    history = Param(Params._dummy(), 'history', 'history')
    model = Param(Params._dummy(), 'model', 'model')
    feature_columns = Param(Params._dummy(), 'feature_columns', 'feature columns')
    label_columns = Param(Params._dummy(), 'label_columns', 'label columns')
    run_id = Param(Params._dummy(), 'run_id',
                   'unique ID for the run that generated this model, if no ID was given by the '
                   'user, defaults to current timestamp at the time of fit()',
                   typeConverter=TypeConverters.toString)
    _metadata = Param(Params._dummy(), '_metadata',
                      'metadata contains the shape and type of input and output')

    def __init__(self):
        super(SparkModelParams, self).__init__()

    @keyword_only
    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def setHistory(self, value):
        return self._set(history=value)

    def getHistory(self):
        return self.getOrDefault(self.history)

    def setModel(self, value):
        return self._set(model=value)

    def getModel(self):
        return self.getOrDefault(self.model)

    def setFeatureColumns(self, value):
        return self._set(feature_columns=value)

    def getFeatureColumns(self):
        return self.getOrDefault(self.feature_columns)

    def setLabelColoumns(self, value):
        return self._set(label_columns=value)

    def getLabelColumns(self):
        return self.getOrDefault(self.label_columns)

    def setRunId(self, value):
        return self._set(run_id=value)

    def getRunId(self):
        return self.getOrDefault(self.run_id)

    # Only for internal use
    def _get_metadata(self):
        return self.getOrDefault(self._metadata)
