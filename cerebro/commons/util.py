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

import inspect
import tensorflow as tf


def fix_huggingface_layer_methods_and_add_to_custom_objects(estimator):
    model = estimator.getModel()
    custom_objects = estimator.getCustomObjects()

    for layer in model.layers:
        if issubclass(type(layer), tf.keras.layers.Layer) and inspect.getmodule(layer).__name__.startswith('transformers.'):
            if not hasattr(layer, 'config'):
                raise RuntimeError('{} layer should have an explicitly set `config` variable.'.format(type(layer)))
            
            patch_hugginface_layer_methods(type(layer))

            if type(layer).__name__ not in custom_objects:
                custom_objects[type(layer).__name__] =  type(layer)

    estimator.setCustomObjects(custom_objects)


def patch_hugginface_layer_methods(cls):
    def patch_init(cls):
        if not hasattr(cls, 'orig_init'):
            from transformers import PretrainedConfig
            cls.orig_init = cls.__init__
            def init(self, config, **kwargs):
                if isinstance(config, dict):
                    config = PretrainedConfig.from_dict(config)
                self.config = config
                cls.orig_init(self, config, **kwargs)
            cls.__init__ = init

    def patch_get_config(cls):
        __class__ = cls
        def get_config(self):
            cfg = super().get_config()
            cfg['config'] = self.config.to_dict()
            return cfg
        cls.get_config = get_config

    patch_get_config(cls)
    patch_init(cls)
