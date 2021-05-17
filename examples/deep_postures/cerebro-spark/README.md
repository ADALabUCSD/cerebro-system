
# Background
We provide an example of how to use `cerebro-spark` to scale the model selection of behavior classification models that were released in the [DeepPostures](https://github.com/ADALabUCSD/DeepPostures/tree/master/MSSE-2021) library. `cerebro-spark` is a Spark-based implementation of model hopper parallelism that can be run on a shared file system such as NFS or HDFS. More details on how to use `cerebro-spark` can be found in the [documentation page](https://adalabucsd.github.io/cerebro-system/).


# Steps
## 1. Preparing the Training Data
1. Use the [DeepPostures](https://github.com/ADALabUCSD/DeepPostures/tree/master/MSSE-2021) library to generate the pre-processed data. This library generates pre-processed data in H5 format. Store these pre-processed files in a shared directory (e.g., NFS mount or HDFS), which is accessible by Spark workers. Pre-processed data is organized by directories each corresponding to a different human subject. Each directory contains multiple files corresponding to each device-wear date.
2. Run the `prep_petastorm_data.py` script to transform the pre-preocessed data in step 1 to Petastorm format, which is the format used by `cerebro-spark`. Change the constant values defined in the main method of the script to match your setting before running the script.
    ~~~
    SPARK_MASTER_URL = 'spark://...' # Change the Spark master URL.
    H5_PRE_PROCESSED_DATA_DIR = 'file://...' # Change pre-processed data input path. Should be accessible from all Spark workers.
    OUTPUT_PATH = 'file:///...' # Change Petastorm output path. Should be accessible from all Spark workers.
    TRAIN_FRACTION = 0.7 # Fraction of train data. Remaining is validation data.
    ~~~

## 2. (Optional) Modify the Workload
1. The workload is defined in the `model_selection.py` script using `cerebro-spark` APIs. There are three main componenets: 1) `search_space` definition, 2) `estimator_gen_fn` function, and 3) `model_selection` object.
2. `search_space` is a dictionary defining the tuning parameters. Currently, it defines 2 different values for learning rate, regularization value, amp factor, and BiLSTM window size, each. You can add more parameters values if you need. More details about these tuning parameters can be found in our [paper](https://github.com/ADALabUCSD/DeepPostures).
3. `estimator_gen_fn` takes in a parameter value instance and initilizes a `cerebro.SparkEstimator` object, which is ready to be trained.
4. `model_selection` object encapsulates the model selection procedure. The script uses `GridSearch`. However, you can also use `RandomSearch` or `TPESearch`.

More details on how to set/define `search_space`, `estimator_gen_fn`, and `model_selection` can be found in `cerebro-spark` [documentation](https://adalabucsd.github.io/cerebro-system/)

## 3. Running the workload
1. You need to first install cerebro on all your Spark workers. This can be done by running the command `pip install cerebro-dl`.
2. Change the constants defined in the main method of the script to match your setting. `DATA_STORE_PATH` should be same as the `OUTPUT_PATH` in the data pre-processing section.
    ~~~
    SPARK_MASTER_URL = 'spark://...' # Change the Spark master URL.
    DATA_STORE_PATH = 'file:///...' # Change data store path. Should be accessible from all Spark workers.
    ~~~
3. Run the script by executing the command: `python model_selection.py`.
