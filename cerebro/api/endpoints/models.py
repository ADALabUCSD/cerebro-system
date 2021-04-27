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
from flask import request
from flask_restplus import Resource
from flask_restplus import abort
from werkzeug.exceptions import BadRequest
from sqlalchemy import and_
from ..restplus import api
from ..serializers import model
from ..parsers import experiment_id_argument
from ...db.dao import Model, Experiment, ParamVal, ParamDef
from ...db import db
from ...commons.constants import *
from .experiments import next_user_friendly_model_id

ns = api.namespace('models', description='Operations related to models')


@ns.route('/')
class ModelsCollection(Resource):
    @api.marshal_list_with(model)
    @api.expect(experiment_id_argument, validate=True)
    def get(self):
        """
        Returns list of models.
        """
        args = experiment_id_argument.parse_args()
        exp_id = args['exp_id']
        return Model.query.filter(Model.exp_id == exp_id).all()

    @api.expect(model)
    def post(self):
        """
        Creates a new model.
        """
        data = request.json
        exp_id = data.get('exp_id')
        num_trained_epochs =  0
        max_train_epochs = data.get('max_train_epochs')
        warm_start_model_id = data.get('warm_start_model_id')

        if warm_start_model_id is not None:
            warm_start_model = Model.query.filter(Model.id == warm_start_model_id).one()
            print(warm_start_model.status, warm_start_model.num_trained_epochs)
            assert warm_start_model.status in [RUNNING_STATUS, COMPLETED_STATUS] and warm_start_model.num_trained_epochs > 0, \
            'Warm start model should be a completed or current running model with atleast 1 epoch trained.'

        exp = Experiment.query.filter(Experiment.id == exp_id).one()
        if exp.status in [FAILED_STATUS, STOPPED_STATUS, COMPLETED_STATUS]:
            raise BadRequest('Experiment is in {} staus. Cannot create new models.'.format(exp.status))

        model_id = next_user_friendly_model_id()
        model_dao = Model(model_id, exp_id, num_trained_epochs, max_train_epochs, warm_start_model_id)

        for pval in data.get('param_vals'):
            name = pval.get('name')
            value = pval.get('value')

            dtype = ParamDef.query.filter(and_(ParamDef.exp_id == exp_id, ParamDef.name == name)).one().dtype

            pval_dao = ParamVal(model_dao.id, name, value, dtype)
            db.session.add(pval_dao)

        db.session.add(model_dao)
        db.session.commit()
        return model_dao.id, 201


@ns.route('/<string:id>')
@api.response(404, 'Model not found.')
class GetModel(Resource):
    
    @api.marshal_with(model)
    def get(self, id):
        """
        Returns a model.
        """
        return Model.query.filter(Model.id == id).one()


@ns.route('/stop/<string:id>')
@api.response(404, 'Model not found.')
@api.response(400, 'Invalid request.')
class StopModel(Resource):
    
    @api.response(204, 'Model successfully stopped.')
    def post(self, id):
        """
        Stops a model.
        """
        model = Model.query.filter(Model.id == id).one()
        if model.status == RUNNING_STATUS:
            model.status = STOPPED_STATUS
            db.session.commit()
            return None, 204
        else:
            abort(400, 'Model cannot be stopped from {} status'.format(model.status))


@ns.route('/resume/<string:id>')
@api.response(404, 'Model not found.')
@api.response(400, 'Invalid request.')
class ResumeModel(Resource):
    
    @api.response(204, 'Model successfully resumed.')
    def post(self, id):
        """
        Resumes a model.
        """
        model = Model.query.filter(Model.id == id).one()
        exp = Experiment.query.filter(Experiment.id == model.exp_id).one()
        if model.status == STOPPED_STATUS and exp.status == RUNNING_STATUS:
            model.status = RUNNING_STATUS
            db.session.commit()
            return None, 204
        else:
            abort(400, 'Model cannot be resumed from {} status'.format(model.status))
