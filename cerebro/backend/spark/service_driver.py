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

import psutil
from six.moves import queue, socketserver

from .service_common import find_port, AckResponse, PingRequest, PingResponse, NoValidAddressesFound, Wire


class TaskHostHashIndicesRequest(object):
    """Request task indices for a given host hash."""

    def __init__(self, host_hash):
        self.host_hash = host_hash


class TaskHostHashIndicesResponse(object):
    def __init__(self, indices):
        self.indices = indices
        """Task indices."""

class AllTaskAddressesRequest(object):
    """Request all task addresses for a given index."""

    def __init__(self, index):
        self.index = index


class AllTaskAddressesResponse(object):
    def __init__(self, all_task_addresses):
        self.all_task_addresses = all_task_addresses
        """Map of interface to list of (ip, port) pairs."""


class RegisterTaskRequest(object):
    def __init__(self, index, task_addresses, host_hash):
        self.index = index
        """Task index."""

        self.task_addresses = task_addresses
        """Map of interface to list of (ip, port) pairs."""

        self.host_hash = host_hash
        """
        Hash of the host that helps to determine which tasks
        have shared memory access to each other.
        """


class SparkDriverService:
    NAME = 'driver service'

    def __init__(self, num_workers, key, nics):
        self._service_name = SparkDriverService.NAME
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

        self._num_workers = num_workers
        self._all_task_addresses = {}
        self._task_addresses_for_driver = {}
        self._task_host_hash_indices = {}
        self._wait_cond = threading.Condition()

        self._spark_job_failed = False

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

    def _filter_by_ip(self, addresses, target_ip):
        for intf, intf_addresses in addresses.items():
            for ip, port in intf_addresses:
                if ip == target_ip:
                    return {intf: [(ip, port)]}
        return {}

    def task_addresses_for_driver(self, index):
        return self._task_addresses_for_driver[index]

    def task_addresses_for_tasks(self, index):
        return self._all_task_addresses[index]

    def task_host_hash_indices(self):
        return self._task_host_hash_indices

    def wait_for_initial_registration(self, timeout):
        self._wait_cond.acquire()
        try:
            while len(self._all_task_addresses) < self._num_workers:
                self._wait_cond.wait(timeout.remaining())
                timeout.check_time_out_for('tasks to start')
        finally:
            self._wait_cond.release()

    def _handle(self, req, client_address):

        if isinstance(req, TaskHostHashIndicesRequest):
            return TaskHostHashIndicesResponse(self._task_host_hash_indices[req.host_hash])

        if isinstance(req, RegisterTaskRequest):
            self._wait_cond.acquire()
            try:
                assert 0 <= req.index < self._num_workers
                self._all_task_addresses[req.index] = req.task_addresses
                # Just use source address for service for fast probing.
                self._task_addresses_for_driver[req.index] = \
                    self._filter_by_ip(req.task_addresses, client_address[0])
                if not self._task_addresses_for_driver[req.index]:
                    # No match is possible if one of the servers is behind NAT.
                    # We don't throw exception here, but will allow the following
                    # code fail with NoValidAddressesFound.
                    print('ERROR: Task {index} declared addresses {task_addresses}, '
                          'but has connected from a different address {source}. '
                          'This is not supported. Is the server behind NAT?'
                          ''.format(index=req.index, task_addresses=req.task_addresses,
                                    source=client_address[0]))
                # Make host hash -> indices map.
                if req.host_hash not in self._task_host_hash_indices:
                    self._task_host_hash_indices[req.host_hash] = []
                self._task_host_hash_indices[req.host_hash].append(req.index)
                self._task_host_hash_indices[req.host_hash].sort()
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()
            return AckResponse()

        if isinstance(req, AllTaskAddressesRequest):
            return AllTaskAddressesResponse(self._all_task_addresses[req.index])

        if isinstance(req, PingRequest):
            return PingResponse(self._service_name, client_address[0])

        raise NotImplementedError(req)


    def notify_spark_job_failed(self):
        self._wait_cond.acquire()
        try:
            self._spark_job_failed = True
        finally:
            self._wait_cond.notify_all()
            self._wait_cond.release()

    def check_for_spark_job_failure(self):
        if self._spark_job_failed:
            raise Exception('Spark job has failed, see the error above.')

    def wait_for_initial_registration(self, timeout):
        self._wait_cond.acquire()
        try:
            while len(self._all_task_addresses) < self._num_workers:
                self.check_for_spark_job_failure()
                self._wait_cond.wait(timeout.remaining())
                timeout.check_time_out_for('Spark tasks to start')
        finally:
            self._wait_cond.release()


class SparkDriverClient:
    def __init__(self, driver_addresses, key, verbose, match_intf=False, probe_timeout=20, retries=3):
        service_name = SparkDriverService.NAME

        # Note: because of retry logic, ALL RPC calls are REQUIRED to be idempotent.
        self._verbose = verbose
        self._service_name = service_name
        self._wire = Wire(key)
        self._match_intf = match_intf
        self._probe_timeout = probe_timeout
        self._retries = retries
        self._addresses = self._probe(driver_addresses)
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
                'Linux.'.format(service_name=service_name, addresses=driver_addresses))

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

    def register_task(self, index, task_addresses, host_hash):
        self._send(RegisterTaskRequest(index, task_addresses, host_hash))

    def all_task_addresses(self, index):
        resp = self._send(AllTaskAddressesRequest(index))
        return resp.all_task_addresses

    def task_host_hash_indices(self, host_hash):
        resp = self._send(TaskHostHashIndicesRequest(host_hash))
        return resp.indices