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


class CerebroEstimator(object):

    def get_model_shapes(self):
        raise NotImplementedError('Abstract class. Method not implemented!')

    def create_model(self, history, run_id, metadata):
        raise NotImplementedError('Abstract class. Method not implemented!')

    def get_model_class(self):
        raise NotImplementedError('Abstract class. Method not implemented!')


class CerebroModel(object):
    """ Wrapper object containing a trained Keras model. Can be used for making predictions on a DataFrame.

    """

    def setCustomObjects(self, value):
        raise NotImplementedError('Abstract class. Method not implemented!')

    def getCustomObjects(self):
        raise NotImplementedError('Abstract class. Method not implemented!')

    def set_output_columns(self, output_columns):
        """
        Sets the output column names

        :param output_columns: List of output column names.
        """
        self.setOutputCols(output_columns)

    def transform(self, df):
        """Transforms the given DataFrame using the trained ML model

        :param df: Input DataFrame
        :return: Transformed DataFrame
        """
        raise NotImplementedError('Abstract class. Method not implemented!')

    def keras(self):
        """ Returns the trained model in Keras format.

            :return: TensorFlow Keras Model
        """
        raise NotImplementedError('Abstract class. Method not implemented!')