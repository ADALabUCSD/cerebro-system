# Copyright 2021 Supun Nakandala, and Arun Kumar. All Rights Reserved.
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

import logging
from werkzeug.exceptions import BadRequest
from flask import request
from flask_restplus import Resource
from ..restplus import api
from ..serializers import experiment
from ..database.dbo import Experiment, ParamDef

from ..database import db

log = logging.getLogger(__name__)

ns = api.namespace('experiments', description='Operations related to experiments')


@ns.route('/')
class ExperimentCollection(Resource):
    @api.marshal_list_with(experiment)
    def get(self):
        """
        Returns list of experiments.
        """
        return Experiment.query.all()


    @api.expect(experiment)
    @api.response(201, 'Experiment successfully created.')
    def post(self):
        """
        Creates a new experiment.
        """

        # TODO: We do not support multi-tenancy. Thus we don't allow users to create more than one active experiment at a time.
        if Experiment.query.filter(Experiment.status.in_(['created', 'running'])).count() > 0:
            raise BadRequest('An experiment is still being run. Please wait until it completes to create a new experiment.')

        data = request.json
        name = data.get('name')
        description = data.get('description')
        feature_columns = data.get('feature_columns')
        label_columns =  data.get('label_columns')
        max_train_epochs = data.get('max_train_epochs')
        training_data_prefix_path =  data.get('training_data_prefix_path')
        executable_entrypoint = data.get('executable_entrypoint')

        exp_dao = Experiment(name, description, feature_columns, label_columns, max_train_epochs, training_data_prefix_path, executable_entrypoint)
        db.session.add(exp_dao)

        for pdef in data.get('param_defs'):
            name = pdef.get('name')
            param_type = pdef.get('param_type')
            values = pdef.get('values') if 'values' in pdef else None
            min_val = pdef.get('min_val') if 'min_val' in pdef else None
            max_val = pdef.get('max_val') if 'max_val' in pdef else None
            count = pdef.get('count') if 'count' in pdef else None
            base = pdef.get('base') if 'base' in pdef else None

            pdef_dao = ParamDef(exp_dao.id, name, param_type, values, min_val, max_val, count, base)
            db.session.add(pdef_dao)

        db.session.commit()        

        return exp_dao.id, 201


@ns.route('/<string:id>')
@api.response(404, 'Experiment not found.')
class ExperimentItem(Resource):
    
    @api.marshal_with(experiment)
    def get(self, id):
        """
        Returns an experiment with all its models.
        """
        return Experiment.query.filter(Experiment.id == id).one()


    @api.response(204, 'Experiment successfully stopped.')
    def delete(self, id):
        """
        Stops experiment.
        """
        raise NotImplementedError()
        # return None, 204
