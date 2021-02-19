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

from flask import request
from flask_restplus import Resource
from ..restplus import api
from ..serializers import experiment

log = logging.getLogger(__name__)

ns = api.namespace('experiments', description='Operations related to experiments')


@ns.route('/')
class ExperimentCollection(Resource):
    @api.marshal_list_with(experiment)
    def get(self):
        """
        Returns list of experiments.
        """
        raise NotImplementedError()

    @api.expect(experiment)
    @api.marshal_with(str, code=201)
    def post(self):
        """
        Creates a new experiment.
        """
        raise NotImplementedError()
        #return None, 201


@ns.route('/<string:id>')
@api.response(404, 'Experiment not found.')
class ExperimentItem(Resource):
    
    @api.marshal_with(experiment)
    def get(self, id):
        """
        Returns an experiment with all its models.
        """
        raise NotImplementedError()


    @api.response(204, 'Experiment successfully deleted.')
    def delete(self, id):
        """
        Deletes experiment.
        """
        raise NotImplementedError()
        # return None, 204
