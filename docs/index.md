---
layout: default
---

``Cerebro`` is a data system for optimized deep learning model selection.


Installation
------------

The best way to install the ``Cerebro`` is via pip.

    pip install -U cerebro

Alternatively, you can git clone and run the provided Makefile script

    git clone https://github.com/ADALabUCSD/cerebro-system.git && cd cerebro-system && make

You MUST be running on **Python >= 3.6** with **Tensorflow >= 2.0** and **Apache Spark >= 2.4**


Getting Started
---------------

Cerebro allows you to perform model selection of your deep neural network directly on an existing Spark DataFrame,
 leveraging Spark’s ability to scale across multiple workers:
 
```python
from cerebro.backend import SparkBackend
from cerebro.keras import SparkEstimator

# datas storage for intermediate data and model artifacts 
from cerebro.storage import LocalStore, HDFSStore

# Model selection/AutoML methods
from cerebro.tune import GridSearch, RandomSearch, HyperOpt 

# Utility functions for specifying the search space
from cerebro.tune import hp_choice, hp_uniform, hp_quniform, hp_loguniform, hp_qloguniform

import tensorflow as tf
from pyspark.sql import SparkSession


spark = SparkSession \
    .builder \
    .appName("Cerebro Example") \
    .getOrCreate()

...

backend = SparkBackend(spark_context=spark.sparkContext, num_proc=3)
store = HDFSStore('/user/username/experiments')

# Define estimator generating function.
# Input: Dictionary containing parameter values
# Output: SparkEstimator 
def estimator_gen_fn(params):
    model = tf.keras.models.Sequential()
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
    
# Define dictionary containing the parameter search space
search_space = {
    'lr': hp_choice([0.01, 0.001, 0.0001]),
    'batchsize': hp_quniform(16, 256, 16)
}

# Instantiate model selection object
hyperopt = HyperOpt(backend=backend, store=store, estimator_gen_fn=estimator_gen_fn, search_space=search_space,
            num_models=30, num_epochs=10, validation=0.25, evaluation_metric='loss',
            feature_column='features', label_column='label', logdir='/tmp/logs')
                  
# Perform model selection                  
hyperopt_output = hyperopt.fit(train_df)

# Inspect model selection results
best_model = hyperopt_output.get_best_model()
best_model_keras = best_model.keras()

pred = best_model_keras.predict([np.ones([1, 692], dtype=np.float32)])

all_models = hyperopt_output.get_all_models()
model_training_metrics = hyperopt_output.get_metrics()

# Perform inference using the best model
output_df = hyperopt_output.transform(test_df)
output_df.select('label', 'label__output').show(n=10)

```

Cerebro hides the complexity of gluing Spark DataFrames to a deep learning training script, reading data into a
format interpretable by the training framework, and distributing the model selection using model hopper parallelism.
 The user only needs to provide a Keras model generating function, define a search space, and pick an AutoML method.

After model selection, Cerebro returns a Transformer representation of the best models. It also contain all the
other models and their training metrics history. The model transformer can be used like any Spark ML transformer to make
 predictions on an input DataFrame, writing them as new columns in the output DataFrame. The best model 
 (and also the other models) can also be converted to Keras format and used in other ways.

The user provided Store object is used to store all model checkpoints, all intermediate representations of the training 
data, and metrics logs (for Tensorboard). Cerebro currently supports stores for HDFS
and local filesystems.


Spark Cluster Setup
-------------------
As deep learning workloads tend to have very different resource requirements
from typical data processing workloads, there are certain considerations
for DL Spark cluster setup.

### GPU training

For GPU training, one approach is to set up a separate GPU Spark cluster
and configure each executor with ``# of CPU cores`` = ``# of GPUs``. This can
be accomplished in standalone mode as follows:

```bash
    $ echo "export SPARK_WORKER_CORES=<# of GPUs>" >> /path/to/spark/conf/spark-env.sh
    $ /path/to/spark/sbin/start-all.sh
```

This approach turns the ``spark.task.cpus`` setting to control # of GPUs
requested per process (defaults to 1).

The ongoing [SPARK-24615](https://issues.apache.org/jira/browse/SPARK-24615) effort aims to
introduce GPU-aware resource scheduling in future versions of Spark.

### CPU training
For CPU training, one approach is to specify the ``spark.task.cpus`` setting
during the training session creation:

```python
    conf = SparkConf().setAppName('training') \
        .setMaster('spark://training-cluster:7077') \
        .set('spark.task.cpus', '16')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

```

This approach allows you to reuse the same Spark cluster for data preparation
and training.


Environment knobs
-----------------

* ``CEREBRO_SPARK_START_TIMEOUT`` - sets the default timeout for Spark tasks to spawn, register, and start running the
 code.  If executors for Spark tasks are scheduled on-demand and can take a long time to start, it may be useful to
  increase this timeout on a system level.