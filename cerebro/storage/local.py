# Copyright 2020 University of California Regents. All Rights Reserved.
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

import pyarrow as pa

from .base import FilesystemStore


class LocalStore(FilesystemStore):
    """Uses the local filesystem as a store of intermediate data and training artifacts."""

    FS_PREFIX = 'file://'

    def __init__(self, prefix_path, *args, **kwargs):
        self._fs = pa.LocalFileSystem()
        super(LocalStore, self).__init__(prefix_path, *args, **kwargs)

    def path_prefix(self):
        return self.FS_PREFIX

    def get_filesystem(self):
        return self._fs

    def get_local_output_dir_fn(self, run_id):
        run_path = self.get_localized_path(self.get_run_path(run_id))

        @contextlib.contextmanager
        def local_run_path():
            if not os.path.exists(run_path):
                try:
                    os.makedirs(run_path, mode=0o755)
                except OSError as e:
                    # Race condition from workers on the same host: ignore
                    if e.errno != errno.EEXIST:
                        raise
            yield run_path

        return local_run_path

    def sync_fn(self, run_id):
        run_path = self.get_localized_path(self.get_run_path(run_id))

        def fn(local_run_path):
            # No-op for LocalStore since the `local_run_path` will be the same as the run path
            assert run_path == local_run_path
        return fn

    @classmethod
    def filesystem_prefix(cls):
        return cls.FS_PREFIX
