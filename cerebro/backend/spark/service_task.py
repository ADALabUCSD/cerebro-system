# Copyright 2020 Supun Nakandala, Yuhao Zhang, and Arun Kumar. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

import socket
import threading
import traceback
import tensorflow as tf
from distutils.version import LooseVersion

import psutil
import pyspark
from six.moves import queue, socketserver

from .service_common import find_port, PingResponse, PingRequest, NoValidAddressesFound, AckResponse, Wire


class SetLocalTaskIndexRequest(object):
    def __init__(self, local_task_index):
        self.local_task_index = local_task_index
        """Local rank of the task"""


class InitDataLoadersRequest(object):
    def __init__(self, store_prefix_path, initialize_data_loaders_fn):
        self.store_prefix_path = store_prefix_path
        self.initialize_data_loaders_fn = initialize_data_loaders_fn


class ExecuteSubEpochRequest(object):
    def __init__(self, sub_epoch_fn, store_prefix_path, train, initial_epoch):
        self.sub_epoch_fn = sub_epoch_fn
        self.store_prefix_path = store_prefix_path
        self.is_train = train
        self.initial_epoch = initial_epoch


class SubEpochCompletedRequest(object):
    """Is command execution finished?"""
    pass


class SubEpochCompletedResponse(object):
    def __init__(self, flag, sub_epoch_result):
        self.flag = flag
        """Yes/no"""

        self.sub_epoch_result = sub_epoch_result
        """RUNNING/FAILED/COMPLETED and sub-epoch result"""


class NotifyInitialRegistrationCompleteRequest(object):
    """Notification that initial task registration has completed."""
    pass


class NotifyWorkloadCompleteRequest(object):
    """Notification that the workload has completed."""
    pass


class SparkTaskService:
    NAME_FORMAT = 'task service #%d'

    def __init__(self, index, key, nics):
        # disabling eager
        # tf.compat.v1.disable_eager_execution()

        service_name = SparkTaskService.NAME_FORMAT % index
        self._index = index
        self._service_name = service_name
        self._wire = Wire(key)
        self._nics = nics
        self._server, _ = find_port(
            lambda addr: socketserver.ThreadingTCPServer(
                addr, self._make_handler()))
        self._port = self._server.socket.getsockname()[1]
        self._addresses = self._get_local_addresses()
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.daemon = True
        self._thread.start()

        self.local_task_index = 0
        self._initial_registration_complete = False
        self._workload_complete = False
        self._wait_cond = threading.Condition()

        self._sub_epoch_thread = None
        self._sub_epoch_status = None

        self._train_readers = {}
        self._val_readers = {}

    def _make_handler(self):
        server = self

        class _Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    req = server._wire.read(self.rfile)
                    resp = server._handle(req, self.client_address)
                    if not resp:
                        raise Exception('Handler did not return a response.')
                    server._wire.write(resp, self.wfile)
                except EOFError:
                    # Happens when client is abruptly terminated, don't want to pollute the logs.
                    pass

        return _Handler

    def _get_local_addresses(self):
        result = {}
        for intf, intf_addresses in psutil.net_if_addrs().items():
            if self._nics and intf not in self._nics:
                continue
            for addr in intf_addresses:
                if addr.family == socket.AF_INET:
                    if intf not in result:
                        result[intf] = []
                    result[intf].append((addr.address, self._port))
        if not result and self._nics:
            raise NoValidAddressesFound(
                'No available network interface found matching user provided interface: {}'.format(self._nics))
        return result

    def addresses(self):
        return self._addresses

    def shutdown(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def get_port(self):
        return self._port

    def wait_for_initial_registration(self, timeout):
        self._wait_cond.acquire()
        try:
            while not self._initial_registration_complete:
                self._wait_cond.wait(timeout.remaining())
                timeout.check_time_out_for('tasks to start')
        finally:
            self._wait_cond.release()

    def wait_for_workload_completion(self, timeout=5):
        self._wait_cond.acquire()
        try:
            while not self._workload_complete:
                self._wait_cond.wait(timeout)
        finally:
            self._wait_cond.release()

    def _handle(self, req, client_address):

        if isinstance(req, InitDataLoadersRequest):
            self._wait_cond.acquire()
            try:
                store_prefix_path = req.store_prefix_path
                if store_prefix_path not in self._train_readers:
                    train_reader, val_reader = req.initialize_data_loaders_fn(self._index)
                    self._train_readers[store_prefix_path] = train_reader
                    self._val_readers[store_prefix_path] = val_reader
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()
            return AckResponse()

        if isinstance(req, SetLocalTaskIndexRequest):
            self.local_task_index = req.local_task_index
            return AckResponse()

        if isinstance(req, ExecuteSubEpochRequest):
            self._wait_cond.acquire()
            try:
                if self._sub_epoch_thread is None or not self._sub_epoch_thread.is_alive():
                    self._sub_epoch_status = None

                    def bg_execute(fn, store_prefix_path, is_train, initial_epoch):
                        try:
                            self._sub_epoch_status = {"status": "RUNNING", "result": None}
                            if is_train:
                                reader = self._train_readers[store_prefix_path]
                            else:
                                reader = self._val_readers[store_prefix_path]
                            func_result = fn(reader, is_train, initial_epoch,
                                             local_task_index=self.local_task_index)
                            self._sub_epoch_status = {"status": "COMPLETED", "result": func_result}
                        except Exception as e:
                            self._sub_epoch_status = {"status": "FAILED", "result": None,
                                                      "error": str(e) + "\n" + traceback.format_exc()}

                    self._sub_epoch_thread = threading.Thread(target=bg_execute, args=(req.sub_epoch_fn, req.store_prefix_path, req.is_train,
                                                                                       req.initial_epoch))
                    self._sub_epoch_thread.start()
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()

            return AckResponse()

        if isinstance(req, SubEpochCompletedRequest):
            self._wait_cond.acquire()
            try:
                terminated = (self._sub_epoch_thread is not None and
                              not self._sub_epoch_thread.is_alive())
            finally:
                self._wait_cond.release()
            return SubEpochCompletedResponse(terminated, self._sub_epoch_status)

        if isinstance(req, NotifyInitialRegistrationCompleteRequest):
            self._wait_cond.acquire()
            try:
                self._initial_registration_complete = True
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()
            return AckResponse()

        if isinstance(req, NotifyWorkloadCompleteRequest):
            self._wait_cond.acquire()
            try:
                self._workload_complete = True
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()
            return AckResponse()

        if isinstance(req, PingRequest):
            return PingResponse(self._service_name, client_address[0])

        raise NotImplementedError(req)

    def _get_resources(self):
        if LooseVersion(pyspark.__version__) >= LooseVersion('3.0.0'):
            from pyspark import TaskContext
            return TaskContext.get().resources()
        return dict()


class SparkTaskClient:
    def __init__(self, index, task_addresses, key, verbose, match_intf=False, probe_timeout=20, retries=3):
        service_name = SparkTaskService.NAME_FORMAT % index
        self._verbose = verbose
        self._service_name = service_name
        self._wire = Wire(key)
        self._match_intf = match_intf
        self._probe_timeout = probe_timeout
        self._retries = retries
        self._addresses = self._probe(task_addresses)
        if not self._addresses:
            raise NoValidAddressesFound(
                'Cerebro was unable to connect to {service_name} on any '
                'of the following addresses: {addresses}.\n\n'
                'One possible cause of this problem is that '
                'Cerebro currently requires every host to have at '
                'least one routable network interface with the same '
                'name across all of the hosts. '
                'You can run \"ifconfig -a\" '
                'on every host and check for the common '
                'routable interface. '
                'To fix the problem, you can rename interfaces on '
                'Linux.'.format(service_name=service_name, addresses=task_addresses))

    def _probe(self, addresses):
        result_queue = queue.Queue()
        threads = []
        for intf, intf_addresses in addresses.items():
            for addr in intf_addresses:
                thread = threading.Thread(target=self._probe_one,
                                          args=(intf, addr, result_queue))
                thread.daemon = True
                thread.start()
                threads.append(thread)
        for t in threads:
            t.join()

        result = {}
        while not result_queue.empty():
            intf, addr = result_queue.get()
            if intf not in result:
                result[intf] = []
            result[intf].append(addr)
        return result

    def _probe_one(self, intf, addr, result_queue):
        for iter in range(self._retries):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self._probe_timeout)
            try:
                sock.connect(addr)
                rfile = sock.makefile('rb')
                wfile = sock.makefile('wb')
                try:
                    self._wire.write(PingRequest(), wfile)
                    resp = self._wire.read(rfile)
                    if resp.service_name != self._service_name:
                        return
                    if self._match_intf:
                        # Interface name of destination and source must match
                        # since `match_intf` is requested.
                        client_intf_addrs = [x.address
                                             for x in psutil.net_if_addrs().get(intf, [])
                                             if x.family == socket.AF_INET]
                        if resp.source_address not in client_intf_addrs:
                            if self._verbose >= 2:
                                # Need to find the local interface name whose
                                # address was visible to the target host's server.
                                resp_intf = ''
                                for key in psutil.net_if_addrs().keys():
                                    key_intf_addrs = [x.address
                                                      for x in psutil.net_if_addrs().get(key, [])]
                                    if resp.source_address in key_intf_addrs:
                                        resp_intf = key
                                        break
                                print('WARNING: Expected to connect the host '
                                      '{addr} using interface '
                                      '{intf}, but reached it on interface '
                                      '{resp_intf}.'.format(
                                    addr=str(addr[0]) + ':' + str(addr[1]),
                                    intf=intf,
                                    resp_intf=resp_intf))
                            return
                    result_queue.put((intf, addr))
                    return
                finally:
                    rfile.close()
                    wfile.close()
            except:
                pass
            finally:
                sock.close()

    def _send_one(self, addr, req):
        for iter in range(self._retries):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect(addr)
                rfile = sock.makefile('rb')
                wfile = sock.makefile('wb')
                try:
                    self._wire.write(req, wfile)
                    resp = self._wire.read(rfile)
                    return resp
                finally:
                    rfile.close()
                    wfile.close()
            except:
                if iter == self._retries - 1:
                    # Raise exception on the last retry.
                    raise
            finally:
                sock.close()

    def _send(self, req):
        # Since all the addresses were vetted, use the first one.
        addr = list(self._addresses.values())[0][0]
        return self._send_one(addr, req)

    def addresses(self):
        return self._addresses

    def notify_initial_registration_complete(self):
        self._send(NotifyInitialRegistrationCompleteRequest())

    def notify_workload_complete(self):
        self._send(NotifyWorkloadCompleteRequest())

    def initialize_data_loaders(self, store_prefix_path, fn):
        self._send(InitDataLoadersRequest(store_prefix_path, fn))

    def execute_sub_epoch(self, fn, store_prefix_path, train=True, initial_epoch=0):
        self._send(ExecuteSubEpochRequest(fn, store_prefix_path, train, initial_epoch))

    def sub_epoch_completed(self):
        return self._send(SubEpochCompletedRequest())

    def set_local_task_index(self, local_task_index):
        return self._send(SetLocalTaskIndexRequest(local_task_index))
