from cerebro.backend import SparkBackend
from cerebro.keras import SparkEstimator

# datas storage for intermediate data and model artifacts.
from cerebro.storage import LocalStore, HDFSStore

# Model selection/AutoML methods.
from cerebro.tune import GridSearch, RandomSearch, TPESearch

# Utility functions for specifying the search space.
from cerebro.tune import hp_choice, hp_uniform, hp_quniform, hp_loguniform, hp_qloguniform

import tensorflow as tf
# tf.config.run_functions_eagerly(True)

from pyspark.sql import SparkSession


spark = SparkSession \
    .builder \
    .appName("Cerebro Iris") \
    .getOrCreate()

...

backend = SparkBackend(spark_context=spark.sparkContext, num_workers=1)
store = LocalStore(prefix_path='/Users/zijian/Desktop/ucsd/cse234/project/cerebro-system/experiments')

from pyspark.ml.feature import OneHotEncoderEstimator

df = spark.read.csv("/Users/zijian/Desktop/ucsd/cse234/project/cerebro-system/Iris_clean.csv", header=True, inferSchema=True)

encoder = OneHotEncoderEstimator(dropLast=False)
encoder.setInputCols(["Species"])
encoder.setOutputCols(["Species_OHE"])

encoder_model = encoder.fit(df)
encoded = encoder_model.transform(df)

feature_columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
label_columns=['Species_OHE']

# Initialize input DataFrames.
# You can download sample dataset from https://apache.googlesource.com/spark/+/master/data/mllib/sample_libsvm_data.txt

train_df, test_df = encoded.randomSplit([0.8, 0.2])

# Define estimator generating function.
# Input: Dictionary containing parameter values
# Output: SparkEstimator
def estimator_gen_fn(params):
    inputs = [tf.keras.Input(shape=(1,)) for col in feature_columns]
    embeddings = [tf.keras.layers.Dense(16, activation=tf.nn.relu)(input) for input in inputs]
    combined = tf.keras.layers.Concatenate()(embeddings)
    output = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(combined)
    model = tf.keras.Model(inputs, output)

#     inputs = tf.keras.Input(shape=(4,))
#     output1 = tf.keras.layers.Dense(16, activation=tf.nn.relu)(inputs)
#     output2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)(output1)
#     output = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(output2)
#     model = tf.keras.Model(inputs, output)

    optimizer = tf.keras.optimizers.Adam(lr=params['lr'])
    loss = 'categorical_crossentropy'

    estimator = SparkEstimator(
        model=model,
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'],
        batch_size=params['batch_size'])

    return estimator

# Define dictionary containing the parameter search space.
search_space = {
    'lr': hp_choice([0.01, 0.001, 0.0001]),
    'batch_size': hp_quniform(16, 64, 16)
}

# Instantiate TPE (Tree of Parzan Estimators a.k.a., HyperOpt) model selection object.
model_selection = TPESearch(
    backend=backend, 
    store=store, 
    estimator_gen_fn=estimator_gen_fn, 
    search_space=search_space,
    num_models=1, 
    num_epochs=10, 
    validation=0.25, 
    evaluation_metric='loss',
    feature_columns=feature_columns,
    label_columns=label_columns
)

# Perform model selection. Returns best model.
model = model_selection.fit(train_df)

# Inspect best model training history.
model_history = model.get_history()

# # Perform inference using the best model and Spark DataFrame.
output_df = model.set_output_columns(['label_predicted']).transform(test_df)
output_df.select('Species', 'label_predicted').show(n=10)

# # Access all models.
# all_models = model.get_all_models()
# all_model_training_history = model.get_all_model_history()

# # Convert the best model to Keras and perform inference using numpy data.
# keras_model = model.keras()
# pred = keras_model.predict([np.ones([1, 692], dtype=np.float32)])
# # Save the keras checkpoint file.
# keras_model.save(ckpt_path)

# # Convert all the model to Keras.
# all_models_keras = [m.keras() for m in all_models]