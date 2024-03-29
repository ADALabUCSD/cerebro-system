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
import os

PETASTORM_HDFS_DRIVER = 'libhdfs'

ARRAY = 'array'
CUSTOM_SPARSE = 'custom_sparse_format'
NOCHANGE = 'nochange'

MIXED_SPARSE_DENSE_VECTOR = 'mixed_sparse_dense_vector'
SPARSE_VECTOR = 'sparse_vector'
DENSE_VECTOR = 'dense_vector'

TOTAL_BUFFER_MEMORY_CAP_GIB = 4
BYTES_PER_GIB = 1073741824

RANDOM_SEED = int(os.environ['CEREBRO_RANDOM_SEED']
                  ) if 'CEREBRO_RANDOM_SEED' in os.environ else 2020
