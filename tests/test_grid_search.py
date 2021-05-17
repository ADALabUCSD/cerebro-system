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
from cerebro.backend import SparkBackend
from cerebro.keras import SparkEstimator
from cerebro.storage import LocalStore
from cerebro.tune import RandomSearch, GridSearch, hp_choice
from pyspark.sql import SparkSession


def estimator_gen_fn(params):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=692, name='features'))
    model.add(tf.keras.layers.Dense(100, input_dim=692))
    model.add(tf.keras.layers.Dense(1, input_dim=100))
    model.add(tf.keras.layers.Activation('sigmoid'))

    optimizer = tf.keras.optimizers.Adam(lr=params['lr'])
    loss = 'binary_crossentropy'

    def transformation_fn(dataset):
        return dataset.shuffle(1000)

    keras_estimator = SparkEstimator(
        model=model,
        optimizer=optimizer,
        loss=loss,
        metrics=['acc'],
        batch_size=10,
        transformation_fn=transformation_fn)

    return keras_estimator


class TestGridSearch(unittest.TestCase):

    def test_grid_search(self):
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

        search_space = {'lr': hp_choice([0.01, 0.001, 0.0001])}

        grid_search = GridSearch(backend, store, estimator_gen_fn, search_space, 1,
                                 validation=0.25, evaluation_metric='loss',
                                 feature_columns=['features'], label_columns=['label'])

        model = grid_search.fit(df)
        output_df = model.transform(df)
        output_df.select('label', 'label__output').show(n=10)

        assert True

    def test_random_search(self):
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


        ######## Random Search ###########
        search_space = {'lr': hp_choice([0.01, 0.001, 0.0001])}

        random_search = RandomSearch(backend, store, estimator_gen_fn, search_space, 3, 1,
                                     validation=0.25,
                                     evaluation_metric='loss',
                                     feature_columns=['features'], label_columns=['label'])
        model = random_search.fit(df)

        output_df = model.transform(df)
        output_df.select('label', 'label__output').show(n=10)

        assert True

    def test_prepare_data(self):
        spark = SparkSession \
            .builder \
            .master("local[3]") \
            .appName("Python Spark SQL basic example") \
            .getOrCreate()

        # Load training data
        df = spark.read.format("libsvm").load("./tests/sample_libsvm_data.txt").repartition(8)
        df.printSchema()

        backend = SparkBackend(spark_context=spark.sparkContext, num_workers=3)
        store = LocalStore('/tmp', train_path='/tmp/train_data', val_path='/tmp/val_data')
        backend.prepare_data(store, df, validation=0.25)

        ######## Random Search ###########
        search_space = {'lr': hp_choice([0.01, 0.001, 0.0001])}

        random_search = RandomSearch(backend, store, estimator_gen_fn, search_space, 3, 1,
                                     validation=0.25,
                                     evaluation_metric='loss',
                                     feature_columns=['features'], label_columns=['label'])
        model = random_search.fit_on_prepared_data()

        output_df = model.transform(df)
        output_df.select('label', 'label__output').show(n=10)

        assert True


if __name__ == "__main__":
    unittest.main()
