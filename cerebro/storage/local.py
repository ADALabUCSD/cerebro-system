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

import contextlib
import errno
import os
import shutil
import pyarrow as pa

from .base import FilesystemStore


class LocalStore(FilesystemStore):
    """Uses the local filesystem as a store of intermediate data and training artifacts (also works with NFS mounted
    remote storage).

    :param prefix_path: Prefix path of the local directory (e.g., /user/test/cerebro).
    :param train_path: (Optional) Path of the directory to store training data. If not specified will default to
        <prefix_path>/train_data
    :param val_path: (Optional) Path of the directory to store validation data. If not specified will default to
        <prefix_path>/val_data
    :param runs_path: (Optional) Path of the directory to store model checkpoints and log. If not specified will default
        to <prefix_path>/runs
    """
    FS_PREFIX = 'file://'

    def __init__(self, prefix_path, train_path=None, val_path=None, runs_path=None, temp_dir=None):
        self._fs = pa.LocalFileSystem()
        self._temp_dir = temp_dir
        super(LocalStore, self).__init__(prefix_path, train_path=train_path, val_path=val_path, runs_path=runs_path)

    def path_prefix(self):
        return self.FS_PREFIX

    def get_filesystem(self):
        return self._fs

    def exists(self, path):

        return os.path.exists(self.get_localized_path(path))

    def _get_filesystem_fn(self):
        return self.get_filesystem

    @classmethod
    def filesystem_prefix(cls):
        return cls.FS_PREFIX

    def move(self, fs, local_path, remote_path):
        remote_path = self.get_localized_path(remote_path)
        os.makedirs(os.path.dirname(remote_path), exist_ok=True)
        shutil.copyfile(local_path, remote_path)
