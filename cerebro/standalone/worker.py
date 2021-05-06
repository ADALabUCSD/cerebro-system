# Copyright 2021 Supun Nakandala. All Rights Reserved.
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

import base64
import sys
import threading
import gc
import traceback
import argparse
from xmlrpc.server import SimpleXMLRPCServer
import dill

data_cache = {}
status_dict = {}

def initialize_worker():
    """
    Initialize the worker by resetting the caches
    :return:
    """
    global data_cache
    global status_dict
    # del data_cache
    # del status_dict
    data_cache = {}
    status_dict = {}
    gc.collect()
    


def execute(exec_id, code_string, params):
    # can execute only one at a time
    """
    :param exec_id:
    :param code_string:
    :param params:
    :return:
    """
    if len([y for y in status_dict.values() if y["status"] == "RUNNING"]) > 0:
        return base64.b64encode(dill.dumps("BUSY"))
    else:
        func = dill.loads(base64.b64decode(code_string))

        def bg_execute(exec_id, func, params):
            """
            :param exec_id:
            :param func:
            :param params:
            """
            try:
                func_result = func(data_cache, *params)
                status_dict[exec_id] = {"status": "COMPLETED", "result": func_result}
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                sys.stdout.flush()
                status_dict[exec_id] = {"status": "FAILED"}

        status_dict[exec_id] = {"status": "RUNNING"}
        thread = threading.Thread(target=bg_execute, args=(exec_id, func, params,))
        thread.start()

        return base64.b64encode(dill.dumps("LAUNCHED"))


def status(exec_id):
    """
    :param exec_id:
    :return:
    """
    if exec_id in status_dict:
        return base64.b64encode(dill.dumps(status_dict[exec_id]))
    else:
        return base64.b64encode(dill.dumps({"status": "INVALID ID"}))


def is_live():
    return True


def main():
    parser = argparse.ArgumentParser(description='Argument parser for generating model predictions.')
    parser.add_argument('--hostname', help='Worker host name', default='0.0.0.0')
    parser.add_argument('--port', help='Worker port', default=7777, type=int)
    args = parser.parse_args()

    print('Starting Cerebro worker on {}:{}'.format(args.hostname, args.port))
    server = SimpleXMLRPCServer((args.hostname, args.port), allow_none=True)

    server.register_function(execute)
    server.register_function(status)
    server.register_function(initialize_worker)
    server.register_function(is_live)
    server.serve_forever()

if __name__ == "__main__":
    main()