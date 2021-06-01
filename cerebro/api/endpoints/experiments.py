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
import sys
import time
import logging
import traceback
from werkzeug.exceptions import BadRequest
from flask import request
from flask_restplus import Resource
from pyspark.sql import SparkSession
from importlib import import_module
from threading import Thread, Lock
from sqlalchemy import and_

from ..restplus import api
from ..serializers import experiment
from ...db.dao import Experiment, ParamDef, Model, ParamVal
from ...db import db
from ..cerebro_server import app
from ...commons.constants import *

from ...backend import SparkBackend
from ...storage import LocalStore, HDFSStore
from ...tune import TPESearch, hp_choice, hp_uniform, hp_quniform, hp_loguniform, hp_qloguniform
from ...tune.grid import HILGridSearch, HILRandomSearch

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
        data = request.json
        name = data.get('name')
        description = data.get('description')
        model_selection_algorithm = data.get('model_selection_algorithm')
        max_num_models = data.get('max_num_models')
        feature_columns = data.get('feature_columns')
        label_columns =  data.get('label_columns')
        max_train_epochs = data.get('max_train_epochs')
        data_store_prefix_path =  data.get('data_store_prefix_path')
        executable_entrypoint = data.get('executable_entrypoint')
        clone_model_id = data.get('clone_model_id')
        warm_start_from_cloned_model = data.get('warm_start_from_cloned_model')

        # Experiment validation
        if model_selection_algorithm in [MS_RANDOM_SEARCH, MS_HYPEROPT_SEARCH]:
            assert max_num_models is not None and max_num_models > 0, '{} should have valid max_num_models value'.format(model_selection_algorithm)

        exp_dao = Experiment(name, description, clone_model_id, warm_start_from_cloned_model, model_selection_algorithm, max_num_models, feature_columns,
                            label_columns, max_train_epochs, data_store_prefix_path, executable_entrypoint)
        db.session.add(exp_dao)

        pdefs = data.get('param_defs')
        for pdef in pdefs:
            name = pdef.get('name')
            param_type = pdef.get('param_type')
            choices = pdef.get('choices')
            min = pdef.get('min')
            max = pdef.get('max')
            q = pdef.get('q')
            dtype = pdef.get('dtype')

            # Parameter validation
            if param_type == HP_CHOICE:
                assert choices is not None, '{} should have non null choices'.format(param_type)
            if param_type in [HP_LOGUNIFORM, HP_QLOGUNIFORM, HP_QUNIFORM, HP_UNIFORM]:
                assert min is not None and max is not None, '{} should have non null min/max values'.format(param_type)
            if param_type in [HP_QLOGUNIFORM, HP_QUNIFORM]:
                assert q is not None, '{} should have non null q value'.format(param_type)

            pdef_dao = ParamDef(exp_dao.id, name, param_type, choices, min, max, q, dtype)
            db.session.add(pdef_dao)


        # Cloning an expriment. Populating remaining hyperparameter values from the cloned model.
        if clone_model_id is not None:
            cloned_model = Model.query.filter(Model.id == clone_model_id).one()
            cloned_model_param_vals = cloned_model.param_vals

            for cparam_val in cloned_model_param_vals:
                param_present = False
                for pdef in pdefs:
                    if cparam_val.name == pdef.name:
                        param_present = True
                        break

                if not param_present:
                    pdef_dao = ParamDef(exp_dao.id, cparam_val.name, HP_CHOICE, [cparam_val.value], None, None, None, cparam_val.dtype)
                    db.session.add(pdef_dao)

        db.session.commit()
        thread = Thread(target=experiment_runner, args=(exp_dao.id, app,))
        thread.start()

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


def experiment_runner(exp_id, app):
    with app.app_context():
        exp_obj = Experiment.query.filter(Experiment.id == exp_id).one()
        
        try:
            # Initial experiment creation.
            if exp_obj.status == CREATED_STATUS:
                search_space = {}
                param_defs = ParamDef.query.filter(ParamDef.exp_id == exp_id).all()
                for param_def in param_defs:
                    param_name = param_def.name
                    param_vals = []

                    if param_def.param_type == HP_CHOICE:
                        param_vals = [x.strip()  for x in param_def.choices.split(',')]
                        if param_def.dtype == DTYPE_FLOAT:
                            param_vals = [float(x) for x in param_vals]
                        elif param_def.dtype == DTYPE_INT:
                            param_vals = [int(x) for x in param_vals]

                        search_space[param_name] = hp_choice(param_vals)
                    else:
                        min_val = param_def.min
                        max_val = param_def.max
                        if param_def.dtype == DTYPE_FLOAT:
                            min_val = float(min_val)
                            max_val = float(max_val)
                        elif param_def.dtype == DTYPE_INT:
                            min_val = int(min_val)
                            max_val = int(max_val)

                        q = None
                        if param_def.param_type in [HP_QUNIFORM, HP_QLOGUNIFORM]:
                            q = param_def.q
                            if param_def.dtype == DTYPE_FLOAT:
                                q = float(q)
                            elif param_def.dtype == DTYPE_INT:
                                q = int(q)

                        if param_def.param_type == HP_UNIFORM:
                            search_space[param_name] = hp_uniform(min_val, max_val)
                        elif param_def.param_type == HP_LOGUNIFORM:
                            search_space[param_name] = hp_loguniform(min_val, max_val)
                        elif param_def.param_type == HP_QUNIFORM:
                            search_space[param_name] = hp_quniform(min_val, max_val, q)
                        elif param_def.param_type == HP_QLOGUNIFORM:
                            search_space[param_name] = hp_qloguniform(min_val, max_val, q)
                        else:
                            raise NotImplementedError('Unsupported hyperparameter type: {}'.format(param_def.param_type))

                backend = app.config['CEREBRO_BACKEND']
                data_store_prefix_path = exp_obj.data_store_prefix_path
                if data_store_prefix_path.startswith('hdfs://'):
                    store = HDFSStore(prefix_path=data_store_prefix_path)
                else:
                    store = LocalStore(prefix_path=data_store_prefix_path)

                if exp_obj.model_selection_algorithm == MS_GRID_SEARCH:
                    model_selection = HILGridSearch(
                        exp_id=exp_obj.id, backend=backend, store=store, estimator_gen_fn=None, search_space=search_space,
                        num_epochs=int(exp_obj.max_train_epochs), db=db, verbose=2 if app.config['DEBUG'] else 0
                    )
                elif exp_obj.model_selection_algorithm == MS_RANDOM_SEARCH:
                    model_selection = HILRandomSearch(
                        exp_id=exp_obj.id, backend=backend, store=store, estimator_gen_fn=None, search_space=search_space,
                        num_models=int(exp_obj.max_num_models), num_epochs=int(exp_obj.max_train_epochs), db=db, verbose=2 if app.config['DEBUG'] else 0
                    )
                elif exp_obj.model_selection_algorithm == MS_HYPEROPT_SEARCH:
                    # TODO
                    raise NotImplementedError()

                model_selection.fit_on_prepared_data()
                
                db.session.refresh(exp_obj)
                exp_obj.status = COMPLETED_STATUS
                db.session.commit()
        except Exception as e:
            logging.error(traceback.format_exc())
            db.session.refresh(exp_obj)
            exp_obj.status = FAILED_STATUS
            exp_obj.exception_message = str(traceback.format_exc())
            db.session.commit()
