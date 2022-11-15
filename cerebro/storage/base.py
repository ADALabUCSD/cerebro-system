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
from __future__ import print_function

import os

import pyarrow.parquet as pq

import inspect


def filter_dict(dict_to_filter, thing_with_kwargs):
    sig = inspect.signature(thing_with_kwargs)
    filter_keys = [param.name for param in sig.parameters.values(
    ) if param.kind == param.POSITIONAL_OR_KEYWORD]
    filtered_dict = {
        filter_key: dict_to_filter[filter_key] for filter_key in filter_keys}
    return filtered_dict


class Store(object):
    """
    Storage layer for intermediate files (materialized DataFrames) and training artifacts (checkpoints, logs).

    Store provides an abstraction over a filesystem (e.g., local vs HDFS) or blob storage database. It provides the
    basic semantics for reading and writing objects, and how to access objects with certain definitions.

    The store exposes a generic interface that is not coupled to a specific DataFrame, model, or runtime. Every run
    of an Estimator should result in a separate run directory containing checkpoints and logs.
    """
    def __init__(self):
        self._train_data_to_key = {}
        self._val_data_to_key = {}

    def is_parquet_dataset(self, path):
        """Returns True if the path is the root of a Parquet dataset."""
        raise NotImplementedError()

    def get_parquet_dataset(self, path):
        """Returns a :py:class:`pyarrow.parquet.ParquetDataset` from the path."""
        raise NotImplementedError()

    def get_train_data_path(self, idx=None):
        """Returns the path to the training dataset."""
        raise NotImplementedError()

    def get_val_data_path(self, idx=None):
        """Returns the path to the validation dataset."""
        raise NotImplementedError()

    def get_test_data_path(self, idx=None):
        """Returns the path to the test dataset."""
        raise NotImplementedError()

    def saving_runs(self):
        """Returns True if run output should be saved during training."""
        raise NotImplementedError()

    def get_runs_path(self):
        """Returns the base path for all runs."""
        raise NotImplementedError()

    def get_run_path(self, run_id):
        """Returns the path to the run with the given ID."""
        raise NotImplementedError()

    def get_checkpoint_path(self, run_id):
        """Returns the path to the checkpoint file for the given run."""
        raise NotImplementedError()

    def get_logs_path(self, run_id):
        """Returns the path to the log directory for the given run."""
        raise NotImplementedError()

    def get_checkpoint_filename(self):
        """Returns the basename of the saved checkpoint file."""
        raise NotImplementedError()

    def get_logs_subdir(self):
        """Returns the subdirectory name for the logs directory."""
        raise NotImplementedError()

    def exists(self, path):
        """Returns True if the path exists in the store."""
        raise NotImplementedError()

    def read(self, path):
        """Returns the contents of the path as bytes."""
        raise NotImplementedError()

    def get_local_output_dir_fn(self, run_id):
        raise NotImplementedError()

    def sync_fn(self, run_id):
        """Returns a function that synchronises given path recursively into run path for `run_id`."""
        raise NotImplementedError()

    def to_remote(self, run_id, dataset_idx=None):
        """Returns a view of the store that can execute in a remote environment without Horoovd deps."""
        attrs = self._remote_attrs(run_id, dataset_idx)

        class RemoteStore(object):
            def __init__(self):
                for name, attr in attrs.items():
                    setattr(self, name, attr)

        return RemoteStore()

    def _remote_attrs(self, run_id, dataset_idx):
        return {
            'train_data_path': self.get_train_data_path(dataset_idx),
            'val_data_path': self.get_val_data_path(dataset_idx),
            'test_data_path': self.get_test_data_path(dataset_idx),
            'runs_path': self.get_runs_path(),
            'run_path': self.get_run_path(run_id),
            'checkpoint_path': self.get_checkpoint_path(run_id),
            'checkpoint_filename': self.get_checkpoint_filename(),
            'get_local_output_dir': self.get_local_output_dir_fn(run_id),
            'get_local_logs_dir': self.get_local_output_dir_fn(run_id),
            'sync': self.sync_fn(run_id),
            'get_last_checkpoint': lambda: self.read(self.get_checkpoint_path(run_id))
        }


class FilesystemStore(Store):
    """Abstract class for stores that use a filesystem for underlying storage."""

    def __init__(self, prefix_path, train_path=None, val_path=None, test_path=None, runs_path=None):
        self.prefix_path = self.get_full_path(prefix_path)
        self._train_path = self._get_full_path_or_default(train_path, 'train_data')
        self._val_path = self._get_full_path_or_default(val_path, 'val_data')
        self._test_path = self._get_full_path_or_default(test_path, 'test_data')
        self._runs_path = self._get_full_path_or_default(runs_path, 'runs')
        super(FilesystemStore, self).__init__()

    def exists(self, path):
        return self.get_filesystem().exists(self.get_localized_path(path))

    def read(self, path):
        with self.get_filesystem().open(self.get_localized_path(path), 'rb') as f:
            return f.read()

    def is_parquet_dataset(self, path):
        try:
            dataset = self.get_parquet_dataset(path)
            return dataset is not None
        except:
            return False

    def get_parquet_dataset(self, path):
        return pq.ParquetDataset(self.get_localized_path(path), filesystem=self.get_filesystem())

    def get_train_data_path(self, idx=None):
        return '{}.{}'.format(self._train_path, idx) if idx is not None else self._train_path

    def get_val_data_path(self, idx=None):
        return '{}.{}'.format(self._val_path, idx) if idx is not None else self._val_path

    def get_test_data_path(self, idx=None):
        return '{}.{}'.format(self._test_path, idx) if idx is not None else self._test_path

    def get_runs_path(self):
        return self._runs_path

    def get_run_path(self, run_id):
        return os.path.join(self.get_runs_path(), run_id)

    def get_checkpoint_path(self, run_id):
        return os.path.join(self.get_run_path(run_id), self.get_checkpoint_filename())

    def get_checkpoint_filename(self):
        return 'checkpoint.h5'

    def get_full_path(self, path):
        if not self.matches(path):
            return self.path_prefix() + path
        return path

    def get_localized_path(self, path):
        if self.matches(path):
            return path[len(self.path_prefix()):]
        return path

    def get_full_path_fn(self):
        prefix = self.path_prefix()

        def get_path(path):
            return prefix + path
        return get_path

    def _get_full_path_or_default(self, path, default_key):
        if path is not None:
            return self.get_full_path(path)
        return self._get_path(default_key)

    def _get_path(self, key):
        return os.path.join(self.prefix_path, key)

    def path_prefix(self):
        raise NotImplementedError()

    def get_filesystem(self):
        raise NotImplementedError()

    @classmethod
    def matches(cls, path):
        return path.startswith(cls.filesystem_prefix())

    @classmethod
    def filesystem_prefix(cls):
        raise NotImplementedError()
