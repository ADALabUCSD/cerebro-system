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


class KerasEstimator(object):

    def get_model_shapes(self):
        raise NotImplementedError('Abstract class. Method not implemented!')

    def create_model(self, history, run_id, metadata):
        raise NotImplementedError('Abstract class. Method not implemented!')

    def get_model_class(self):
        raise NotImplementedError('Abstract class. Method not implemented!')


class KerasModel(object):

    def setCustomObjects(self, value):
        raise NotImplementedError('Abstract class. Method not implemented!')

    def getCustomObjects(self):
        raise NotImplementedError('Abstract class. Method not implemented!')

    def transform(self, df):
        raise NotImplementedError('Abstract class. Method not implemented!')