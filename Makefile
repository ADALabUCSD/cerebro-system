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


all: install

build: clean
	python3 setup.py sdist bdist_wheel

install: clean
	python3 setup.py install

publish: clean build
	python3 -m pip install --user --upgrade twine
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
	#twine upload dist/*

gen-docs:
	cd docs-src && rm -rf build/* && make html && cp -r build/html/* ../docs && cd ..

clean:
	rm -rf build dist cerebro.egg-info .cache .pytest_cache
