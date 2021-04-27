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
import werkzeug
import shutil
from flask import request
from flask_restplus import Resource
from ..restplus import api
from ..parsers import upload_parser
from ...db import db
from ..cerebro_server import app


ns = api.namespace('scripts', description='Operations related to uploading model creation scripts')


@ns.route('/upload')
class ScriptsUpload(Resource):
    @api.response(201, 'Category successfully created.')
    @api.expect(upload_parser, validate=False)
    def post(self):
        """
        Upload new model creation script.
        """
        args = upload_parser.parse_args()
        upload_file = args['file']
        upload_file.save(os.path.join(app.config['SCRIPTS_DIR'], upload_file.filename))
        return None, 201


@ns.route('/')
class ScriptsDelete(Resource):

    @api.response(204, 'All scripts successfully deleted.')
    def delete(self):
        """
        Empties the scripts directory.
        """
        shutil.rmtree(app.config['SCRIPTS_DIR'])
        os.makedirs(app.config['SCRIPTS_DIR'])
        return None, 204
