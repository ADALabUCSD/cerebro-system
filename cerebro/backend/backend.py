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


class Backend(object):
    """Interface for remote execution of the distributed training function.
    """

    def _num_workers(self):
        """Returns the number of workers to use for training."""
        raise NotImplementedError()

    def initialize_workers(self):
        """Initialize workers"""
        raise NotImplementedError()

    def initialize_data_loaders(self, store, schema_fields):
        """Initialize data loaders"""
        raise NotImplementedError()

    def train_for_one_epoch(self, models, store, feature_col, label_col, is_train=True):
        """
        Takes a set of Keras models and trains for one epoch. If is_train is False, validation is performed
         instead of training.
        :param models:
        :param store: single store object common for all models or a dictionary of store objects indexed by model id.
        :param feature_col: single list of feature columns common for all models or a dictionary of feature lists indexed by model id.
        :param label_col: single list of label columns common for all models or a dictionary of label lists indexed by model id.
        :param is_train:
        """
        raise NotImplementedError()

    def teardown_workers(self):
        """Teardown workers"""
        raise NotImplementedError()

    def prepare_data(self, store, dataset, validation, compress_sparse=False, verbose=2):
        """
        Prepare data by writing out into persistent storage
        :param store:
        :param dataset:
        :param validation:
        :param compress_sparse:
        :param verbose:
        """
        raise NotImplementedError()

    def get_metadata_from_parquet(self, store, label_columns=['label'], feature_columns=['features']):
        """
        Get metadata from existing data in the persistent storage
        :param store:
        :param label_columns:
        :param feature_columns:
        """
        raise NotImplementedError()