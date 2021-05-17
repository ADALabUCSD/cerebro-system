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

import unittest

import tensorflow as tf
from pyspark.sql import SparkSession

from cerebro.backend import SparkBackend
from cerebro.storage import LocalStore

from cerebro.keras import SparkEstimator
from cerebro.tune import TPESearch, hp_choice, hp_uniform, hp_quniform, hp_qloguniform


class TestTPE(unittest.TestCase):
    def test_tpe(self):
        spark = SparkSession \
            .builder \
            .master("local[3]") \
            .appName("Python Spark SQL basic example") \
            .getOrCreate()

        # Load training data
        df = spark.read.format("libsvm").load("./tests/sample_libsvm_data.txt").repartition(8)
        df.printSchema()

        backend = SparkBackend(spark_context=spark.sparkContext, num_workers=3)
        store = LocalStore('/tmp')

        def estimator_gen_fn(params):
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Input(shape=692, name='features'))
            model.add(tf.keras.layers.Dense(100, input_dim=692))
            model.add(tf.keras.layers.Dense(1, input_dim=100))
            model.add(tf.keras.layers.Activation('sigmoid'))

            optimizer = tf.keras.optimizers.Adam(lr=params['lr'])
            loss = 'binary_crossentropy'

            keras_estimator = SparkEstimator(
                model=model,
                optimizer=optimizer,
                loss=loss,
                metrics=['acc'],
                batch_size=10)

            return keras_estimator

        search_space = {
            'lr': hp_choice([0.01, 0.001, 0.0001]),
            'dummy1': hp_uniform(0, 100),
            'dummy2': hp_quniform(0, 100, 1),
            'dummy3': hp_qloguniform(0, 100, 1),
        }

        hyperopt = TPESearch(backend=backend, store=store, estimator_gen_fn=estimator_gen_fn, search_space=search_space,
                            num_models=3, num_epochs=1, validation=0.25, evaluation_metric='loss',
                            feature_columns=['features'], label_columns=['label'], verbose=2)

        model = hyperopt.fit(df)
        output_df = model.transform(df)
        output_df.select('label', 'label__output').show(n=10)

        assert True


if __name__ == "__main__":
    unittest.main()
