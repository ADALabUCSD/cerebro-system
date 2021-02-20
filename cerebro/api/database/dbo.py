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

################### Experiment ##################
class Experiment(db.Model):
    id = db.Column(db.String(32), primary_key=True)
    name = db.Column(db.String(32))
    description = db.Column(db.String(512))
    creation_time = db.Column(db.DateTime)
    last_update_time = db.Column(db.DateTime)
    status = db.Column(db.String(32))
    feature_columns = db.Column(db.String(512))
    label_columns = db.Column(db.String(512))
    max_train_epochs = db.Column(db.Integer())
    training_data_prefix_path = db.Column(db.String(512))
    executable_entrypoint = db.Column(db.String(512))

    param_defs = db.relationship('ParamDef', backref='experiment', lazy='dynamic')
    models = db.relationship('Model', backref='model', lazy='dynamic')


    def __init__(self, name, description, feature_columns, label_columns, max_train_epochs, training_data_prefix_path, executable_entrypoint):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.creation_time = datetime.utcnow()
        self.last_update_time = self.creation_time
        self.status = 'created'
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.max_train_epochs = max_train_epochs
        self.training_data_prefix_path = training_data_prefix_path
        self.executable_entrypoint = executable_entrypoint

    def __repr__(self):
        return '<Experiment %r>' % self.id


class Model(db.Model):
    id = db.Column(db.String(32), primary_key=True)
    exp_id = db.Column(db.String(32), db.ForeignKey('experiment.id'))
    creation_time = db.Column(db.DateTime)
    last_update_time = db.Column(db.DateTime)
    status = db.Column(db.String(32))
    num_trained_epochs = db.Column(db.Integer())
    max_train_epochs = db.Column(db.Integer())

    param_vals = db.relationship('ParamVal', backref='model', lazy='dynamic')

    def __init__(self, exp_id, num_trained_epochs, max_train_epochs):
        self.id = str(uuid.uuid4())
        self.exp_id = exp_id        
        self.creation_time = datetime.utcnow()
        self.last_update_time = self.creation_time
        self.status = 'created'
        self.num_trained_epochs = num_trained_epochs
        self.max_train_epochs = max_train_epochs

    def __repr__(self):
        return '<Model %r>' % self.id


class ParamDef(db.Model):
    name = db.Column(db.String(32), primary_key=True)
    exp_id = db.Column(db.String(32), db.ForeignKey('experiment.id'))
    param_type = db.Column(db.String(32))
    values = db.Column(db.String(512))
    min_val = db.Column(db.Float())
    max_val = db.Column(db.Float())
    count = db.Column(db.Integer())
    base = db.Column(db.Integer())

    def __init__(self, exp_id, name, param_type, values=None, min_val=0, max_val=0, count=0, base=0):
        self.exp_id = exp_id
        self.name = name
        self.param_type = param_type
        self.values = values
        self.min_val = min_val
        self.max_val = max_val
        self.count = count
        self.base = base

    def __repr__(self):
        return '<ParamDef %r>' % self.id


class ParamVal(db.Model):
    name = db.Column(db.String(32), db.ForeignKey('param_def.name'), primary_key=True)
    model_id = db.Column(db.String(32), db.ForeignKey('model.id'))
    value = db.Column(db.String(32))
    
    def __init__(self, model_id, name, value):
        self.model_id = model_id
        self.name = name
        self.value = value

    def __repr__(self):
        return '<ParamVal %r>' % self.id
