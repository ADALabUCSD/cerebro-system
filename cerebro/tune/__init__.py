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

"""
cerebro.tune
============
This module contains several AutoML/Model Selection procedures and utility functions for defining search spaces.
"""

from .base import hp_choice, hp_uniform, hp_quniform, hp_loguniform, hp_qloguniform
from .grid import GridSearch, RandomSearch, ModelSelectionResult
from .tpe import TPESearch


hp_choice.__module__ = "cerebro.tune"
hp_uniform.__module__ = "cerebro.tune"
hp_quniform.__module__ = "cerebro.tune"
hp_loguniform.__module__ = "cerebro.tune"
hp_qloguniform.__module__ = "cerebro.tune"

GridSearch.__module__ = "cerebro.tune"
RandomSearch.__module__ = "cerebro.tune"
TPESearch.__module__ = "cerebro.tune"

ModelSelectionResult.__module__ = "cerebro.tune"
