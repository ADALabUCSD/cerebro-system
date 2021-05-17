from pyspark.sql import SparkSession

import os

from cerebro.backend import SparkBackend
from cerebro.keras import SparkEstimator
from cerebro.storage import LocalStore
from cerebro.tune import RandomSearch, GridSearch, hp_choice

import tensorflow as tf


col_names = ['id', 'time', 'non_wear', 'sleeping', 'label', 'data']


def estimator_gen_fn(params):

    amp_factor = params['amp_factor']
    l2_reg = params['l2_reg']
    lr = params['lr']
    win_size = params['win_size'] # in mins
    pre_processed_data_granularity = 10 # seconds
    
    assert 60%pre_processed_data_granularity == 0
    win_size = win_size * int(60/pre_processed_data_granularity)

    input_data = tf.keras.layers.Input(shape=(win_size, 100, 3, 1), name='data')
    x = input_data

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32*amp_factor, [5, 3], strides=(2, 1), data_format='channels_last', padding= "same", activation='relu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64*amp_factor, [5, 1], strides=(2, 1), data_format='channels_last', padding= "same", activation='relu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128*amp_factor, [5, 1], strides=(2, 1), data_format='channels_last', padding= "same", activation='relu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256*amp_factor, [5, 1], strides=(2, 1), data_format='channels_last', padding= "same", activation='relu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256*amp_factor, [5, 1], strides=(2, 1), data_format='channels_last', padding= "same", activation='relu'))(x)
    
    x = tf.keras.layers.Permute((1, 4, 2, 3))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), merge_mode='concat')(x)
    logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2))(x)

    optimizer = tf.keras.optimizers.Adam(lr=lr)
    loss = 'categorical_crossentropy'

    model = tf.keras.models.Model(inputs=[input_data], outputs=[logits])

    # Adding regularization.
    regularizer = tf.keras.regularizers.l2(float(l2_reg))
    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # TensorFlow dataset transformation function that transforms 10s accelerometer data into windows that
    # can be fed to the BiLSTM. It reads accelrometer data in sorted order (sort by subject_id, timestamp).
    def transformation_fn(dataset):
        dataset = dataset.map(lambda row: list([getattr(row, col_name) for col_name in col_names]))
        dataset = dataset.window(win_size, drop_remainder=True).flat_map(lambda *args: zip(*args).batch(6*7))
        dataset = dataset.map(lambda *args: {col_name: args[i] for i, col_name in enumerate(col_names)})

        def filter_fn(window):
            # check the window is from the same subject
            same_subject = len((tf.unique(window['id']))[0]) == 1
            # check the window does not have any non-wear data
            no_non_wear = tf.math.reduce_max(window['non_wear']) == 0
            # check the window does not have any sleeping data
            no_sleep = tf.math.reduce_max(window['sleeping']) == 0
            # check the window does not have any missing labels. If a label is missing it is set to -1.
            no_missing_label = tf.math.reduce_min(window['label']) != -1
            # ensure the window is time sorted.
            time_orderd = tf.math.reduce_mean(tf.math.abs(tf.sort(window['time']) - window['time']))  == 0
            
            return same_subject and no_non_wear and no_sleep and no_missing_label and time_orderd


        dataset = dataset.filter(filter_fn)
        return dataset.map(lambda row: {'data' : row['data'], 'label' : tf.one_hot(row['label'], 2)})


    keras_estimator = SparkEstimator(
        model=model,
        optimizer=optimizer,
        loss=loss,
        metrics=['acc'],
        batch_size=128,
        transformation_fn=transformation_fn)

    return keras_estimator


def main():
    SPARK_MASTER_URL = 'spark://...' # Change the Spark master URL.
    DATA_STORE_PATH = 'file:///...' # Change data store path. Should be accessible from all Spark workers.
    
    spark = SparkSession \
            .builder \
            # Change the Spark Master URL
            .master(SPARK_MASTER_URL) \
            .appName("Deep Postures Example") \
            .getOrCreate()

    backend = SparkBackend(spark_context=spark.sparkContext, num_workers=1)
    store = LocalStore(DATA_STORE_PATH, train_path=os.path.join(DATA_STORE_PATH, 'train'), val_path=os.path.join(DATA_STORE_PATH, 'valid'))

    search_space = {
                        'lr': hp_choice([0.001, 0.0001]),
                        'l2_reg': hp_choice([0.001, 0.0001]),
                        'win_size': hp_choice([7, 9]),
                        'amp_factor': hp_choice([2, 4])
                }

    model_selection = GridSearch(backend, store, estimator_gen_fn, search_space, 10, evaluation_metric='loss',
                        feature_columns=['id', 'time', 'non_wear', 'sleeping', 'label', 'data'], label_columns=['label'])
    model = model_selection.fit_on_prepared_data()

if __name__ == "__main__":
    main()