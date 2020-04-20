# Copyright 2020 University of California Regents. All Rights Reserved.
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
from cerebro.keras import CerebroSparkEstimator
from cerebro.storage import LocalStore
from cerebro.tune import HyperOpt, hp_choice, hp_uniform


class TestHyperOpt(unittest.TestCase):
    def test_grid_search(self):
        spark = SparkSession \
            .builder \
            .master("local[3]") \
            .appName("Python Spark SQL basic example") \
            .getOrCreate()

        # Load training data
        df = spark.read.format("libsvm").load("./tests/sample_libsvm_data.txt").repartition(8)
        df.printSchema()

        backend = SparkBackend(spark_context=spark.sparkContext, num_proc=3)
        store = LocalStore('/tmp')

        def estimator_gen_fn(params):
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(100, input_dim=692))
            model.add(tf.keras.layers.Dense(1, input_dim=100))
            model.add(tf.keras.layers.Activation('sigmoid'))

            optimizer = tf.keras.optimizers.Adam(lr=params['lr'])
            loss = 'binary_crossentropy'

            keras_estimator = CerebroSparkEstimator(
                model=model,
                optimizer=optimizer,
                loss=loss,
                metrics=['acc'],
                batch_size=10)

            return keras_estimator

        search_space = {
            'lr': hp_choice([0.01, 0.001, 0.0001]),
            'dummy': hp_uniform(0, 100)
        }

        hyperopt = HyperOpt(backend, store, estimator_gen_fn, search_space, 3, 1,
                                 validation=0.25, evaluation_metric='loss',
                                 feature_column='features', label_column='label', logdir='/tmp/logs')

        model = hyperopt.fit(df)
        output_df = model.transform(df)
        output_df.select('label', 'label__output').show(n=10)


if __name__ == "__main__":
    unittest.main()
