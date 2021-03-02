import tensorflow as tf
from cerebro.keras import SparkEstimator

def estimator_gen_fn(params):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(100, input_dim=692))
    model.add(tf.keras.layers.Dense(1, input_dim=100))
    model.add(tf.keras.layers.Activation('sigmoid'))

    optimizer = tf.keras.optimizers.Adam(lr=params['learning_rate'])
    loss = 'binary_crossentropy'

    keras_estimator = SparkEstimator(
        model=model,
        optimizer=optimizer,
        loss=loss,
        metrics=['acc'],
        batch_size=params['batch_size'])

    return keras_estimator
