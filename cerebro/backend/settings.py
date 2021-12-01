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


class Settings(object):

    def __init__(self, verbose=0, key=None, timeout=None, num_workers=None, nics=None,
                 disk_cache_size_bytes=10*1024*1024*1024, data_readers_pool_type='process', num_data_readers=4, polling_period=2):
        """
        :param verbose: level of verbosity
        :param key: used for encryption of parameters passed across the hosts
        :param timeout: has to finish all the checks before this timeout runs out.
        :param num_workers: number of Cerebro processes
        :param disk_cache_size_bytes: Size of the disk data cache in GBs (default 10GB).
        :param data_readers_pool_type: Data readers pool type ('process' or 'thread')
        :param num_data_readers: Number of data readers
        :param nics: specify the NICs to be used for tcp network communication.
        """
        self.verbose = verbose
        self.key = key
        self.timeout = timeout
        self.num_workers = num_workers
        self.nics = nics
        self.polling_period = polling_period
        self.disk_cache_size_bytes = disk_cache_size_bytes
        self.data_readers_pool_type = data_readers_pool_type
        self.num_data_readers = num_data_readers
