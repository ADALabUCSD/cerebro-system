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

class CerebroEstimatorParams(object):

    def setModel(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getModel(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setHyperParams(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getHyperParams(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setStore(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getStore(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setLoss(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getLoss(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setLossWeights(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getLossWeights(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setSampleWeightCol(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getSampleWeightCol(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setMetrics(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getMetrics(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setFeatureCols(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getFeatureCols(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setLabelCols(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getLabelCols(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setValidation(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getValidation(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setCallbacks(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getCallbacks(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setBatchSize(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getBatchSize(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setEpochs(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getEpochs(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setVerbose(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getVerbose(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setShufflingBufferSize(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getShufflingBufferSize(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setOptimizer(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getOptimizer(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setRunId(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getRunId(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setTransformationFn(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getTransformationFn(self):
        raise NotImplementedError('Abstract class. Method not implement!')


class CerebroModelParams(object):

    def setHistory(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getHistory(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setModel(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getModel(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setFeatureColumns(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getFeatureColumns(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setLabelColoumns(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getLabelColumns(self):
        raise NotImplementedError('Abstract class. Method not implement!')

    def setRunId(self, value):
        raise NotImplementedError('Abstract class. Method not implement!')

    def getRunId(self):
        raise NotImplementedError('Abstract class. Method not implement!')
