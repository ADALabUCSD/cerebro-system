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
import os
import re
import shutil
import tempfile

import pyarrow as pa
from pyarrow import fs

from .base import FilesystemStore
from .base import filter_dict



class HDFSStore(FilesystemStore):
    """Uses HDFS as a store of intermediate data and training artifacts.

    Initialized from a `prefix_path` that can take one of the following forms:

    1. "hdfs://namenode01:8020/user/test/Cerebro"
    2. "hdfs:///user/test/Cerebro"
    3. "/user/test/Cerebro"

    :param prefix_path: Prefix path of the local directory (e.g., /user/test/cerebro).
    :param train_path: (Optional) Path of the directory to store training data. If not specified will default to
        <prefix_path>/train_data
    :param val_path: (Optional) Path of the directory to store validation data. If not specified will default to
        <prefix_path>/val_data
    :param runs_path: (Optional) Path of the directory to store model checkpoints and log. If not specified will default
        to <prefix_path>/runs
    :param temp_dir: (Optional) Temp directory on the local machine.
    :param host: (Optional) NameNode hostname.
    :param port: (Optional) NameNode port.
    :param user: (Optional) Username connecting to HDFS.
    :param kerb_ticket: (Optional) Path to Kerberos ticket cache.
    :param driver: (Optional) Driver to be used for HDFS communication (default 'libhdfs')
    :param extra_conf: (Optional) Extra Key/Value pairs for config; Will override any hdfs-site.xml properties
    """

    FS_PREFIX = 'hdfs://'
    URL_PATTERN = '^(?:(.+://))?(?:([^/:]+))?(?:[:]([0-9]+))?(?:(.+))?$'

    def __init__(self, prefix_path, train_path=None, val_path=None, runs_path=None, temp_dir=None,
                 host=None, port=None, user=None, kerb_ticket=None,
                 driver='libhdfs', extra_conf=None):
        self._temp_dir = temp_dir

        prefix, url_host, url_port, path, path_offset = self.parse_url(
            prefix_path)
        self._check_url(prefix_path, prefix, path)
        self._url_prefix = prefix_path[:path_offset] if prefix else self.FS_PREFIX

        host = host or url_host or 'default'
        port = port or url_port or 0
        self._hdfs_kwargs = dict(host=host,
                                 port=port,
                                 user=user,
                                 kerb_ticket=kerb_ticket,
                                 driver=driver,
                                 extra_conf=extra_conf)
        self._hdfs = self._get_filesystem_fn()()

        super(HDFSStore, self).__init__(prefix_path,
                                        train_path=train_path, val_path=val_path, runs_path=runs_path)

    def parse_url(self, url):
        match = re.search(self.URL_PATTERN, url)
        prefix = match.group(1)
        host = match.group(2)

        port = match.group(3)
        if port is not None:
            port = int(port)

        path = match.group(4)
        path_offset = match.start(4)
        return prefix, host, port, path, path_offset

    def exists(self, path):
        return self.get_filesystem().exists(self.get_localized_path(path))

    def path_prefix(self):
        return self._url_prefix

    def get_filesystem(self):
        return self._hdfs

    def _get_filesystem_fn(self):
        hdfs_kwargs = self._hdfs_kwargs

        hdfs_kwargs = filter_dict(hdfs_kwargs, pa.hdfs.connect)

        def fn():
            return pa.hdfs.connect(**hdfs_kwargs)

        return fn

    def _check_url(self, url, prefix, path):
        print('_check_url: {}'.format(prefix))
        if prefix is not None and prefix != self.FS_PREFIX:
            raise ValueError('Mismatched HDFS namespace for URL: {}. Found {} but expected {}'
                             .format(url, prefix, self.FS_PREFIX))

        if not path:
            raise ValueError('Failed to parse path from URL: {}'.format(url))

    @classmethod
    def filesystem_prefix(cls):
        return cls.FS_PREFIX

    def move(self, hdfs, local_path, hdfs_path):
        with open(local_path, 'rb') as f:
            hdfs.upload(hdfs_path, f)
