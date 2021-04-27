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
import traceback

from flask_restplus import Api
from sqlalchemy.orm.exc import NoResultFound

api = Api(version='1.0', title='Cerebro REST API')

@api.errorhandler
def default_error_handler(e):
    message = 'An unhandled exception occurred.'
    print(message)
    return {'message': message}, 500

@api.errorhandler(NoResultFound)
def database_not_found_error_handler(e):
    print(traceback.format_exc())
    return {'message': 'A database result was required but none was found.'}, 404
