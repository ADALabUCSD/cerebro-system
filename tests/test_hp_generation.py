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

import unittest

import numpy as np
from cerebro.tune import hp_choice, hp_uniform, hp_quniform, hp_loguniform, hp_qloguniform


class TestHpGen(unittest.TestCase):
    def test_hp_choice(self):
        options = [['x', 'y', 'z'], [1.0, 2.0, 22.0]]
        for option in options:
            temp = hp_choice(option)
            for _ in range(10):
                sv = temp.sample_value()
                if sv not in option:
                    assert False

        assert True

    def test_hp_uniform(self):
        minV = 0
        maxV = 100
        temp = hp_uniform(minV, maxV)
        for _ in range(10):
            if not (minV <= temp.sample_value() <= maxV):
                assert False

        assert True

    def test_hp_quniform(self):
        minV = 0
        maxV = 100
        q = 3
        temp = hp_quniform(minV, maxV, q)
        for _ in range(10):
            sv = temp.sample_value()
            if not (minV <= sv <= maxV):
                assert False
            elif sv % q != 0:
                assert False

        assert True

    def test_hp_loguniform(self):
        minV = -4
        maxV = -1
        temp = hp_loguniform(minV, maxV)
        for _ in range(10):
            sv = temp.sample_value()
            if not (np.power(0.1, -1*minV) <= sv <= np.power(0.1, -1*maxV)):
                assert False

        assert True

    def test_hp_qloguniform(self):
        minV = -4
        maxV = -1
        q = 0.0005
        temp = hp_qloguniform(minV, maxV, q)
        for _ in range(10):
            sv = temp.sample_value()
            if not (np.power(0.1, -1*minV) <= sv <= np.power(0.1, -1*maxV)):
                assert False
            # elif sv % q != 0: # faces numerical precision issues
            #     assert False

        assert True


if __name__ == "__main__":
    unittest.main()
