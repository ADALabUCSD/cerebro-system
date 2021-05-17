import os
import h5py
import subprocess
import numpy as np
import pandas as pd

from cerebro.backend import SparkBackend
from cerebro.storage import LocalStore

from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, LongType


import random
random.seed(2021)


def load_h5(h5_path):
    fnames = [name for name in os.listdir(h5_path) if not name.startswith('.')]
    fnames.sort()

    dfs = []
    for fname in fnames:
        h5_file = h5py.File(os.path.join(h5_path, fname), 'r')

        df = pd.DataFrame({
            'id': ['AuID004' for _ in range(h5_file['time'].shape[0])],
            'time': h5_file['time'][:].astype('int64'),
            'data': [h5_file['data'][i][:,:].astype('float32') for i in range(h5_file['data'].shape[0])],
            'non_wear': h5_file['non_wear'],
            'sleeping': h5_file['sleeping'],
            'label': h5_file['label']
        })
        dfs.append(df)

    return pd.concat(dfs).values.tolist()


def main():
    SPARK_MASTER_URL = 'spark://...' # Change the Spark master URL.
    H5_PRE_PROCESSED_DATA_DIR = 'file://...' # Change pre-processed data input path. Should be accessible from all Spark workers.
    OUTPUT_PATH = 'file:///...' # Change Petastorm output path. Should be accessible from all Spark workers.
    TRAIN_FRACTION = 0.7 # Fraction of train data. Remaining is validation data.
    
    ROW_GROUP_SIZE_MB = 512 # Size of Parquet row group size.
    NUM_PARTITIONS = 100 # Number of Parquet partitions for train and val data each.
    
    spark = SparkSession \
            .builder \
            .master(SPARK_MASTER_URL) \
            .appName("Deep Postures Example - Petastorm Data Generation") \
            .getOrCreate()

    input_data = []
    if H5_PRE_PROCESSED_DATA_DIR.startswith('hdfs://'):
        args = "hdfs dfs -ls "+dir_in+" | awk '{print $8}'"
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

        s_output, s_err = proc.communicate()
        input_data = ['hdfs://'+ path for path in s_output.split()]
    elif H5_PRE_PROCESSED_DATA_DIR.startswith('file://'):
        for dirname in os.listdir(H5_PRE_PROCESSED_DATA_DIR):
            if not os.path.join(H5_PRE_PROCESSED_DATA_DIR, dirname).startswith('.')
            input_data.append(str(os.path.join(H5_PRE_PROCESSED_DATA_DIR, dirname)))
    else:
        raise Exception('Unsupported file system in: {}'.format(H5_PRE_PROCESSED_DATA_DIR))

    random.shuffle(input_data)
    n_train = int(len(input_data) * TRAIN_FRACTION)
    train_data = input_data[:n_train]
    val_data = input_data[n_train:]

    backend = SparkBackend(spark_context=spark.sparkContext)
    store = LocalStore(OUTPUT_PATH, train_path=os.path.join(OUTPUT_PATH, 'train_data'), val_path=os.path.join(OUTPUT_PATH, 'val_data'))
    
    schema = Unischema('schema', [
        UnischemaField('id', np.string_, (), ScalarCodec(StringType()), False),
        UnischemaField('time', np.int64, (), ScalarCodec(LongType()), False),
        UnischemaField('data', np.float32, (100, 3), NdarrayCodec(), False),
        UnischemaField('non_wear', np.int32, (), ScalarCodec(IntegerType()), False),
        UnischemaField('sleeping', np.int32, (), ScalarCodec(IntegerType()), False),
        UnischemaField('label', np.int32, (), ScalarCodec(IntegerType()), False)
    ])

    with materialize_dataset(spark, os.path.join(output_url, 'train_data'), schema, ROW_GROUP_SIZE_MB):
        rdd=spark.sparkContext.parallelize(train_data)
        rdd = rdd.flatMap(lambda x: load_h5(x)).map(lambda item: {'id': item[0], 'time':item[1], 'data':item[2], 'non_wear':item[3], 'sleeping':item[4], 'label':item[5]})
        rdd =  rdd.map(lambda x: dict_to_spark_row(schema, x)) 
        
        df = spark.createDataFrame(rdd, schema=schema.as_spark_schema())
        df.orderBy("id","time").coalesce(NUM_PARTITIONS).write.mode('overwrite').parquet(os.path.join(output_url, 'train_data'))


    with materialize_dataset(spark, os.path.join(output_url, 'val_data'), schema, ROW_GROUP_SIZE_MB):
        rdd=spark.sparkContext.parallelize(val_data)
        rdd = rdd.flatMap(lambda x: load_h5(x)).map(lambda item: {'id': item[0], 'time':item[1], 'data':item[2], 'non_wear':item[3], 'sleeping':item[4], 'label':item[5]})
        rdd =  rdd.map(lambda x: dict_to_spark_row(schema, x)) 
        
        df = spark.createDataFrame(rdd, schema=schema.as_spark_schema())
        df.orderBy("id","time").coalesce(NUM_PARTITIONS).write.mode('overwrite').parquet(os.path.join(output_url, 'val_data'))

if __name__ == "__main__":
    main()