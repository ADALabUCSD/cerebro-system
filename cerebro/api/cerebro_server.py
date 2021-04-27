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
import os
import sys
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
import argparse
import datetime
import signal
from threading import Thread, Event

from pyspark.sql import SparkSession
from ..backend import SparkBackend

from flask import Flask, Blueprint
from .restplus import api
from ..db import db
from ..db.dao import *

from ..tune.daemon import sub_epoch_scheduler

app = Flask(__name__)

from .endpoints.experiments import ns as experiments_namespace
from .endpoints.models import ns as models_namespace
from .endpoints.scripts import ns as scripts_namespace


def configure_app(flask_app, args):
    flask_app.config['SERVER_NAME'] = args.server_url
    flask_app.config['SPARK_MASTER_URL'] = args.spark_master_url
    flask_app.config['NUM_WORKERS'] = args.num_workers
    flask_app.config['TEMP_DATA_DIR'] = args.temp_data_dir
    flask_app.config['SCRIPTS_DIR'] = os.path.join(args.temp_data_dir, 'scripts')

    if not os.path.exists(args.temp_data_dir):
        os.makedirs(args.temp_data_dir)

    if not os.path.exists(flask_app.config['SCRIPTS_DIR']):
        os.makedirs(flask_app.config['SCRIPTS_DIR'])
    

    flask_app.config['SQLALCHEMY_DATABASE_URI'] = args.database_uri
    flask_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = args.swagger_ui_doc_expansion
    flask_app.config['RESTPLUS_VALIDATE'] = not args.no_restplus_validation
    flask_app.config['RESTPLUS_MASK_SWAGGER'] = args.restplus_mask_swagger
    flask_app.config['ERROR_404_HELP'] = args.restplus_error_404_help


def initialize_app(flask_app, args):
    configure_app(flask_app, args)
    blueprint = Blueprint('api', __name__, url_prefix='/api')
    api.init_app(blueprint)

    # Add namespaces
    api.add_namespace(experiments_namespace)
    api.add_namespace(scripts_namespace)
    api.add_namespace(models_namespace)

    flask_app.register_blueprint(blueprint)
    db.init_app(flask_app)


def main():
    parser = argparse.ArgumentParser(description='Argument parser for Cerebro API server.')

    parser.add_argument('--server-url', help='API server URL.', default='localhost:8888')
    parser.add_argument('--spark-master-url', help='Spark master URL.', default='local[3]')
    parser.add_argument('--num-workers', help='Cerebro number of workers.', default=3)
    parser.add_argument('--database-uri', help='Database URI.', default='sqlite://')
    parser.add_argument('--temp-data-dir', help='Temp data directory.', default='/tmp')
    parser.add_argument('--inter-epoch-wait-time', help='Time delay between epoch schedulings', type=int, default=5)
    parser.add_argument('--swagger-ui-doc-expansion', help='Swagger UI doc expansion model.', default='list')
    parser.add_argument('--no-restplus-validation', help='No RESTPlus validation.', default=False, action='store_true')
    parser.add_argument('--restplus-mask-swagger', help='Whether to mask swagger.', default=False, action='store_true')
    parser.add_argument('--restplus-error-404-help', help='Output 404 error help.', default=False, action='store_true')
    parser.add_argument('--debug', help='Run in debug mode.', default=False, action='store_true')
    args = parser.parse_args()

    initialize_app(app, args)
    print('>>>>> Starting development server at http://{}/api/ <<<<<'.format(app.config['SERVER_NAME']))
    
    with app.app_context():
        db.create_all()

    #intiate the model selection daemon
    sys.path.append(app.config['SCRIPTS_DIR'])
    spark = SparkSession.builder.appName("Cerebro Backend").master(app.config['SPARK_MASTER_URL']).getOrCreate()
    backend = SparkBackend(spark_context=spark.sparkContext, num_workers=app.config['NUM_WORKERS'])
    
    # initialize backend and data loaders
    if args.debug: print('CEREBRO => Time: {}, Initializing Workers'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    backend.initialize_workers()
    app.config['CEREBRO_BACKEND'] = backend
    
    sub_epoch_scheduler_thread = Thread(target=sub_epoch_scheduler, args=(app, db, backend, args.inter_epoch_wait_time, 2 if args.debug == True else 0,))
    sub_epoch_scheduler_thread.start()

    app.run(debug=args.debug)
    sub_epoch_scheduler_thread.join()


if __name__ == "__main__":
    main()
