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
                 disk_cache_size_bytes=10*1024*1024*1024, input_queue_num_proc=1, max_input_queue_size=10, polling_period=2):
        """
        :param verbose: level of verbosity
        :type verbose: int
        :param key: used for encryption of parameters passed across the hosts
        :type key: str
        :param timeout: has to finish all the checks before this timeout runs
        out.
        :type timeout: Cerebro.run.common.util.timeout.Timeout
        :param num_workers: number of Cerebro processes (-np)
        :type num_workers: int
        :param disk_cache_size_bytes: Size of the disk data cache in GBs (default 10GB).
        :type disk_cache_size_bytes: int
        :param max_input_queue_size: Used for input generator input. Maximum size for the generator queue (defaule 10).
        :type max_input_queue_size: int
        :param input_queue_num_proc: Maximum number of processes to spin up when using process-based threading for data loading (default 1).
        :type input_queue_num_proc: int
        :param nics: specify the NICs to be used for tcp network communication.
        :type nics: string
        """
        self.verbose = verbose
        self.key = key
        self.timeout = timeout
        self.num_workers = num_workers
        self.nics = nics
        self.polling_period = polling_period
        self.disk_cache_size_bytes = disk_cache_size_bytes
        self.input_queue_num_proc = input_queue_num_proc
        self.max_input_queue_size = max_input_queue_size
