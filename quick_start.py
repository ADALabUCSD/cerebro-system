from cerebro.backend import SparkBackend
from cerebro.keras import SparkEstimator

# datas storage for intermediate data and model artifacts.
from cerebro.storage import LocalStore, HDFSStore

# Model selection/AutoML methods.
from cerebro.tune import GridSearch, RandomSearch, TPESearch

# Utility functions for specifying the search space.
from cerebro.tune import hp_choice, hp_uniform, hp_quniform, hp_loguniform, hp_qloguniform

import tensorflow as tf
from pyspark.sql import SparkSession


spark = SparkSession \
    .builder \
    .appName("Cerebro Example") \
    .getOrCreate()

...

backend = SparkBackend(spark_context=spark.sparkContext, num_workers=1)
store = LocalStore(prefix_path='/Users/zijian/Desktop/ucsd/cse234/project/cerebro-system/experiments')


# Initialize input DataFrames.
# You can download sample dataset from https://apache.googlesource.com/spark/+/master/data/mllib/sample_libsvm_data.txt
df = spark.read.format("libsvm").load("/Users/zijian/Desktop/ucsd/cse234/project/cerebro-system/sample_libsvm_data.txt").repartition(8)
train_df, test_df = df.randomSplit([0.8, 0.2])

# Define estimator generating function.
# Input: Dictionary containing parameter values
# Output: SparkEstimator
def estimator_gen_fn(params):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=692, name='features'))
    model.add(tf.keras.layers.Dense(100, input_dim=692))
    model.add(tf.keras.layers.Dense(1, input_dim=100))
    model.add(tf.keras.layers.Activation('sigmoid'))

    optimizer = tf.keras.optimizers.Adam(lr=params['lr'])
    loss = 'binary_crossentropy'

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
    'batch_size': hp_quniform(16, 256, 16)
}

# Instantiate TPE (Tree of Parzan Estimators a.k.a., HyperOpt) model selection object.
model_selection = TPESearch(backend=backend, store=store, estimator_gen_fn=estimator_gen_fn, search_space=search_space,
            num_models=30, num_epochs=10, validation=0.25, evaluation_metric='loss',
            feature_columns=['features'], label_columns=['label'])

# Perform model selection. Returns best model.
model = model_selection.fit(train_df)

# Inspect best model training history.
model_history = model.get_history()

# Perform inference using the best model and Spark DataFrame.
output_df = model.set_output_columns(['label_predicted']).transform(test_df)
output_df.select('label', 'label_predicted').show(n=10)

# Access all models.
all_models = model.get_all_models()
all_model_training_history = model.get_all_model_history()

# Convert the best model to Keras and perform inference using numpy data.
keras_model = model.keras()
pred = keras_model.predict([np.ones([1, 692], dtype=np.float32)])
# Save the keras checkpoint file.
keras_model.save(ckpt_path)

# Convert all the model to Keras.
all_models_keras = [m.keras() for m in all_models]
