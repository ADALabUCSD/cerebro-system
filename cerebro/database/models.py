# Copyright 2021 Supun Nakandala and Arun Kumar. All Rights Reserved.
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
import uuid
from datetime import datetime
from . import db


class ParamDef(db.Model):
    id = db.Column(db.String(32), primary_key=True)
    name = db.Column(db.String(32))
    param_type = db.Column(db.String(32))
    values = db.Column(db.String(512))
    min_val = db.Column(db.Float())
    max_val = db.Column(db.Float())
    count = db.Column(db.Integer())
    base = db.Column(db.Integer())

    def __init__(self, name, param_type, values=None, min_val=0, max_val=0, count=0, base=0):
        self.id = str(uuid.uuid4())
        self.name = name
        self.param_type = param_type
        # FIXME: Find a better way to store list of strings
        self.values = ",".join(values)
        self.min_val = min_val
        self.max_val = max_val
        self.count = count
        self.base = base


class ParamVal(db.Model):
    id = db.Column(db.String(32), primary_key=True)
    name = db.Column(db.String(32))
    value = db.Column(db.String(32))
    value_type = db.Column(db.String(32))
    
    def __init__(self, name, value, value_type):
        self.id = str(uuid.uuid4())
        self.name = name
        self.value = value
        self.value_type = value_type

