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
import logging
import traceback
from werkzeug.exceptions import BadRequest
from flask import request
from flask_restplus import Resource
from pyspark.sql import SparkSession
from importlib import import_module
from threading import Thread

from ..restplus import api
from ..serializers import experiment
from ...db.dao import Experiment, ParamDef
from ...db import db
from ..cerebro_server import app
from ...commons.constants import *

from ...backend import SparkBackend
from ...storage import LocalStore, HDFSStore
from ...tune import GridSearch, RandomSearch, TPESearch, hp_choice, hp_uniform, hp_quniform, hp_loguniform, hp_qloguniform

ns = api.namespace('experiments', description='Operations related to experiments')

def experiment_daemon(exp_id, app):
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

                    search_space[param_name] = hp_choice(param_vals)

                data_store_prefix_path = exp_obj.data_store_prefix_path

                if data_store_prefix_path.startswith('hdfs://'):
                    store = HDFSStore(prefix_path=data_store_prefix_path)
                else:
                    store = LocalStore(prefix_path=data_store_prefix_path)

                spark = SparkSession.builder.appName("Cerebro Experiment: {}".format(exp_id)).master(app.config['SPARK_MASTER_URL']).getOrCreate()
                backend = SparkBackend(spark_context=spark.sparkContext, num_workers=app.config['NUM_WORKERS'])

                sys.path.append(app.config['SCRIPTS_DIR'])
                mod, f = exp_obj.executable_entrypoint.split(':')
                mod = import_module(mod)
                estimator_gen_fn = getattr(mod, f)
            
                if exp_obj.model_selection_algorithm == MS_GRID_SEARCH:
                    model_selection = GridSearch(
                        backend=backend, store=store, estimator_gen_fn=estimator_gen_fn, search_space=search_space,
                        num_epochs=int(exp_obj.max_train_epochs), feature_columns=[x.strip() for x in exp_obj.feature_columns.split(',')],
                        label_columns=[x.strip() for x in exp_obj.label_columns.split(',')], verbose=2 if app.config['DEBUG'] else 0
                    )
                elif exp_obj.model_selection_algorithm == MS_RANDOM_SEARCH:
                    model_selection = RandomSearch(
                        backend=backend, store=store, estimator_gen_fn=estimator_gen_fn, search_space=search_space,
                        num_models=int(exp_obj.max_num_models), num_epochs=int(exp_obj.max_train_epochs),
                        feature_columns=[x.strip() for x in exp_obj.feature_columns.split(',')],
                        label_columns=[x.strip() for x in exp_obj.label_columns.split(',')], verbose=2 if app.config['DEBUG'] else 0
                    )
                elif exp_obj.model_selection_algorithm == MS_HYPEROPT_SEARCH:
                    # FIXME: Parallelism is hard-coded here
                    model_selection = TPESearch(
                        backend=backend, store=store, estimator_gen_fn=estimator_gen_fn, search_space=search_space,
                        num_models=int(exp_obj.max_num_models), num_epochs=int(exp_obj.max_train_epochs),
                        feature_columns=[x.strip() for x in exp_obj.feature_columns.split(',')],
                        label_columns=[x.strip() for x in exp_obj.label_columns.split(',')], parallelism= 2*app.config['NUM_WORKERS'], verbose=2 if app.config['DEBUG'] else 0
                    )

                exp_obj.status = RUNNING_STATUS
                db.session.commit()
                
                model_selection.fit_on_prepared_data()

                exp_obj.status = COMPLETED_STATUS
                db.session.commit()
        except Exception as e:
            logging.error(traceback.format_exc())

            exp_obj.status = FAILED_STATUS
            db.session.commit()


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
        if Experiment.query.filter(Experiment.status.in_([CREATED_STATUS, RUNNING_STATUS])).count() > 0:
            raise BadRequest('An experiment is still being run. Please wait until it completes to create a new experiment.')

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

        # Experiment validation
        if model_selection_algorithm in [MS_RANDOM_SEARCH, MS_HYPEROPT_SEARCH]:
            assert max_num_models is not None, '{} should have non max_num_models value'.format(model_selection_algorithm)

        exp_dao = Experiment(name, description, model_selection_algorithm, max_num_models, feature_columns, label_columns, max_train_epochs,
            data_store_prefix_path, executable_entrypoint)
        db.session.add(exp_dao)

        for pdef in data.get('param_defs'):
            name = pdef.get('name')
            param_type = pdef.get('param_type')
            choices = pdef.get('choices')
            min = pdef.get('min')
            max = pdef.get('max')
            q = pdef.get('q')

            # Parameter validation
            if param_type == HP_CHOICE:
                assert choices is not None, '{} should have non null choices'.format(param_type)
            if param_type in [HP_LOGUNIFORM, HP_QLOGUNIFORM, HP_QUNIFORM, HP_UNIFORM]:
                assert min is not None and max is not None, '{} should have non null min/max values'.format(param_type)
            if param_type in [HP_QLOGUNIFORM, HP_QUNIFORM]:
                assert q is not None, '{} should have non null q value'.format(param_type)

            pdef_dao = ParamDef(exp_dao.id, name, param_type, choices, min, max, q)
            db.session.add(pdef_dao)

        db.session.commit()        

        thread = Thread(target=experiment_daemon, args=(exp_dao.id, app,))
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
