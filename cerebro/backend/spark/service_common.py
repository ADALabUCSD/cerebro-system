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

import random
import struct

import cloudpickle

from .. import secret


class PingRequest(object):
    pass


class NoValidAddressesFound(Exception):
    pass


class PingResponse(object):
    def __init__(self, service_name, source_address):
        self.service_name = service_name
        """Service name that responded to this ping."""
        self.source_address = source_address
        """Source IP address that was visible to the service."""


class AckResponse(object):
    """Used for situations when the response does not carry any data."""
    pass


# Given server factory, find a usable port
def find_port(server_factory):
    min_port = 1024
    max_port = 65536
    num_ports = max_port - min_port
    start_port = random.randrange(0, num_ports)
    for port_offset in range(num_ports):
        try:
            port = min_port + (start_port + port_offset) % num_ports
            addr = ('', port)
            server = server_factory(addr)
            return server, port
        except Exception as e:
            pass

    raise Exception('Unable to find a port to bind to.')



class Wire(object):
    """
    Used for serialization/deserialization of objects over the wire.
    We use HMAC to protect services from unauthorized use. The key used for
    the HMAC digest is distributed by Open MPI and Spark.
    The objects are serialized using cloudpickle. Serialized objects become
    the body of the message.
    Structure of the message is as follows:
    - HMAC digest of the body (32 bytes)
    - length of the body (4 bytes)
    - body
    """
    def __init__(self, key):
        self._key = key

    def write(self, obj, wfile):
        message = cloudpickle.dumps(obj)
        digest = secret.compute_digest(self._key, message)
        wfile.write(digest)
        # Pack message length into 8-byte integer.
        wfile.write(struct.pack('l', len(message)))
        wfile.write(message)
        wfile.flush()

    def read(self, rfile):
        digest = rfile.read(secret.DIGEST_LENGTH)
        # Unpack message length into 8-byte integer.
        message_len = struct.unpack('l', rfile.read(8))[0]
        message = rfile.read(message_len)
        if not secret.check_digest(self._key, message, digest):
            raise Exception('Security error: digest did not match the message.')
        return cloudpickle.loads(message)

