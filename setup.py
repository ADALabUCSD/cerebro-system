# Copyright 2020 Supun Nakandala, Yuhao Zhang, and Arun Kumar. All Rights Reserved.
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

from distutils.core import setup
from setuptools import find_packages
import os
this = os.path.dirname(__file__)

with open(os.path.join(this, "requirements.txt"), "r") as f:
    requirements = [_ for _ in [_.strip("\r\n ")
                                for _ in f.readlines()] if _ is not None]

packages = find_packages()
assert packages

# read version from the package file.
version_str = '1.1.1'
with (open(os.path.join(this, 'cerebro/__init__.py'), "r")) as f:
    line = [_ for _ in [_.strip("\r\n ")
                                for _ in f.readlines()] if _.startswith("__version__")]
    if len(line) > 0:
        version_str = line[0].split('=')[1].strip('" ')

setup(
    name='cerebro-dl',
    version=version_str,
    description="Resource-efficient Deep Learning Model Selection on Data Systems",
    long_description_content_type='text/markdown',
    license='Apache License 2.0',
    author='Supun Nakandala',
    author_email='snakanda@eng.ucsd.edu',
    url='https://github.com/ADALabUCSD/cerebro-system',
    packages=packages,
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'cpu': ['tensorflow>=2.3'],
        'gpu': ['tensorflow-gpu>=2.3'],
    },
    entry_points={
        'console_scripts': [
            'cerebro-api = cerebro.api.cerebro_server:main',
            'cerebro-standalone-worker = cerebro.standalone.worker:main',
        ],
    },
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',

        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License'],
)