# Copyright 2020 Supun Nakandala, Yuhao Zhang, and Arun Kumar. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

from __future__ import absolute_import

import datetime
import numpy as np
import pyarrow as pa
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.ml.linalg import DenseVector, SparseVector, Vector, VectorUDT
from pyspark.sql.types import ArrayType, BinaryType, BooleanType, FloatType, DoubleType, \
    IntegerType, LongType, NullType, StringType
try:
    # Spark 3.0 moved to a pandas submodule
    from pyspark.sql.pandas.types import from_arrow_type
except ImportError:
    from pyspark.sql.types import from_arrow_type

from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from petastorm.codecs import ScalarCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.fs_utils import FilesystemResolver
from .. import constants


def data_type_to_str(dtype):
    if dtype == VectorUDT:
        return 'Vector'
    elif dtype == IntegerType:
        return 'Int'
    elif dtype == StringType:
        return 'String'
    elif dtype == FloatType:
        return 'Float'
    elif dtype == BinaryType:
        return 'Binary'
    elif dtype == DoubleType:
        return 'Double'
    elif dtype == LongType:
        return 'Long'
    elif dtype == BooleanType:
        return 'Boolean'
    else:
        raise ValueError('Unrecognized data type: {}'.format(dtype))


def numpy_type_to_str(dtype):
    if dtype == np.int32:
        return 'Int'
    elif dtype == np.float32:
        return 'Float'
    elif dtype == np.uint8:
        return 'Binary'
    elif dtype == np.float64:
        return 'Double'
    elif dtype == np.int64:
        return 'Long'
    elif dtype == np.bool:
        return 'Boolean'
    else:
        raise ValueError('Cannot convert numpy data type to Spark string: {}'.format(dtype))


def spark_scalar_to_python_type(dtype):
    if dtype == IntegerType:
        return int
    elif dtype == StringType:
        return str
    elif dtype == FloatType:
        return float
    elif dtype == DoubleType:
        return float
    elif dtype == LongType:
        return int
    elif dtype == BooleanType:
        return bool
    elif dtype == BinaryType:
        return bytes
    else:
        raise ValueError('cannot convert Spark data Type {} to native python type'.format(dtype))


def pyarrow_to_spark_data_type(dtype):
    # PySpark will interpret list types as Arrays, but for ML applications we want to default to
    # treating these as DenseVectors.
    if pa.types.is_list(dtype):
        return DenseVector
    return type(from_arrow_type(dtype))


def spark_to_petastorm_type(dtype):
    if dtype == VectorUDT or dtype == SparseVector or dtype == DenseVector:
        return np.float64
    elif dtype == ArrayType:
        return np.float64
    elif dtype == IntegerType:
        return np.int32
    elif dtype == StringType:
        return np.string_
    elif dtype == FloatType:
        return np.float32
    elif dtype == BinaryType:
        return np.uint8
    elif dtype == DoubleType:
        return np.float64
    elif dtype == LongType:
        return np.int64
    elif dtype == BooleanType:
        return np.bool
    else:
        raise ValueError('Unrecognized data type: {}'.format(dtype))


def petastorm_unischema_shape(shape):
    if shape == 1:
        return ()
    else:
        return (shape,)


def petastorm_unischema_codec(shape, type):
    if shape == 1:
        return ScalarCodec(type())
    else:
        return NdarrayCodec()


def _get_col_info(df):
    """
    Infer the type and shape of all the columns.

    NOTE: This function processes the entire DataFrame, and can therefore be very expensive to run.

    TODO(travis): Only run this if user sets compress_sparse param, otherwise convert all to Array.
    """

    def get_meta(row):
        row_dict = row.asDict()
        row_schema = []
        for col_name, data_col in row_dict.items():
            dtype = type(data_col)
            if isinstance(data_col, DenseVector):
                # shape and size of dense vector are the same
                shape = size = data_col.array.shape[0]
            elif isinstance(data_col, SparseVector):
                # shape is the total size of vector
                shape = data_col.size
                # size is the number of nonzero elements in the sparse vector
                size = data_col.indices.shape[0]
            elif isinstance(data_col, list):
                shape = size = len(data_col)
            elif isinstance(data_col, type(None)):
                # Python 2.7 compat: NoneType is not pickleable
                # see: https://bugs.python.org/issue6477
                dtype = NullType
                shape = size = 1
            else:
                shape = size = 1
            row_schema.append((col_name, ({dtype}, {shape}, {size})))
        return row_schema

    def merge(x, y):
        x_dtypes, x_shapes, x_sizes = x
        y_dtypes, y_shapes, y_sizes = y
        dtypes = x_dtypes | y_dtypes
        shapes = x_shapes | y_shapes
        sizes = x_sizes | y_sizes
        return dtypes, {min(shapes), max(shapes)}, {min(sizes), max(sizes)}

    raw_col_info_list = df.rdd.flatMap(get_meta).reduceByKey(merge).collect()

    all_col_types = {}
    col_shapes = {}
    col_max_sizes = {}

    for col_info in raw_col_info_list:
        col_name, col_meta = col_info
        dtypes, shapes, sizes = col_meta

        all_col_types[col_name] = dtypes
        col_shapes[col_name] = shapes
        col_max_sizes[col_name] = sizes

    for col in df.schema.names:
        # All rows in every column must have the same shape
        shape_set = col_shapes[col]
        if len(shape_set) != 1:
            raise ValueError(
                'Column {col} does not have uniform shape. '
                'shape set: {shapes_set}'.format(col=col, shapes_set=shape_set))
        col_shapes[col] = shape_set.pop()

        # All rows in every column must have the same size unless they have SparseVectors
        sizes = col_max_sizes[col]
        if len(sizes) > 1 and not (SparseVector in all_col_types[col]):
            raise ValueError(
                'Rows of column {col} have varying sizes. This is only allowed if datatype is '
                'SparseVector or a mix of Sparse and DenseVector.'.format(col=col))
        col_max_sizes[col] = max(sizes)

    return all_col_types, col_shapes, col_max_sizes


def _get_metadata(df):
    """
    Infer the type and shape of all the columns and determines if what intermediate format they
    need to be converted to in case they are a vector.
    """
    all_col_types, col_shapes, col_max_sizes = _get_col_info(df)

    metadata = dict()
    for field in df.schema.fields:
        col = field.name
        col_types = all_col_types[col].copy()

        if DenseVector in col_types:
            # If a col has DenseVector type (whether it is mixed sparse and dense vector or just
            # DenseVector), convert all of the values to dense vector
            is_sparse_vector_only = False
            spark_data_type = DenseVector
            convert_to_target = constants.ARRAY
        elif SparseVector in col_types:
            # If a col has only sparse vectors, convert all the data into custom dense vectors
            is_sparse_vector_only = True
            spark_data_type = SparseVector
            convert_to_target = constants.CUSTOM_SPARSE
        else:
            is_sparse_vector_only = False
            spark_data_type = type(field.dataType)
            convert_to_target = constants.NOCHANGE

        # Explanation of the fields in metadata
        #     dtype:
        #
        #     spark_data_type:
        #         The spark data type from dataframe schema: type(field.dataType). If column has
        #         mixed SparseVector and DenseVector we categorize it as DenseVector.
        #
        #     is_sparse_vector_only:
        #         If all the rows in the column were sparse vectors.
        #
        #     shape:
        #         Determines the shape of the data in the spark dataframe. It is useful for sparse
        #         vectors.
        #
        #     intermediate_format:
        #         Specifies if the column need to be converted to a different format so that
        #         petastorm can read it. It can be one of ARRAY, CUSTOM_SPARSE, or NOCHANGE. It is
        #         required because petastorm cannot read DenseVector and SparseVectors. We need to
        #         identify these types and convert them to petastorm compatible type of array.

        metadata[col] = {'spark_data_type': spark_data_type,
                         'is_sparse_vector_only': is_sparse_vector_only,
                         'shape': col_shapes[col],
                         'intermediate_format': convert_to_target,
                         'max_size': col_max_sizes[col]}

    return metadata


def to_petastorm_fn(schema_cols, metadata):
    ARRAY = constants.ARRAY
    CUSTOM_SPARSE = constants.CUSTOM_SPARSE

    # Convert Spark Vectors into arrays so Petastorm can read them
    def to_petastorm(row):
        from pyspark import Row

        converted = {}
        for col in schema_cols:
            col_data = row[col]
            if isinstance(col_data, Vector):
                intermediate_format = metadata[col]['intermediate_format'] if metadata else ARRAY
                if intermediate_format == ARRAY:
                    converted[col] = col_data.toArray().tolist()
                elif intermediate_format == CUSTOM_SPARSE:
                    # Currently petastorm does not support reading pyspark sparse vector. We put
                    # the indices and values into one array. when consuming the data, we re-create
                    # the vector from this format.
                    size = len(col_data.indices)
                    padding_zeros = 2 * (metadata[col]['max_size'] - len(col_data.indices))

                    converted[col] = np.concatenate(
                        (np.array([size]), col_data.indices, col_data.values,
                         np.zeros(padding_zeros))).tolist()

        if converted:
            row = row.asDict().copy()
            row.update(converted)
        return Row(**row)

    return to_petastorm


def _has_vector_column(df):
    for field in df.schema.fields:
        if isinstance(field.dataType, VectorUDT):
            return True
    return False


def _get_dataset_info(dataset, dataset_id, path):
    total_rows = 0
    total_byte_size = 0
    for piece in dataset.pieces:
        metadata = piece.get_metadata()
        total_rows += metadata.num_rows
        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            total_byte_size += row_group.total_byte_size

    if total_rows == 0:
        raise ValueError('No rows found in {} dataset: {}'.format(dataset_id, path))

    if total_byte_size == 0:
        raise ValueError('No data found in {} dataset: {}'.format(dataset_id, path))

    if total_rows > total_byte_size:
        raise ValueError('Found {} bytes in {} rows; {} dataset may be corrupted.'
                         .format(total_byte_size, total_rows, dataset_id))

    return total_rows, total_byte_size


def get_simple_meta_from_parquet(store, schema_cols, dataset_idx=None):
    train_data_path = store.get_train_data_path(dataset_idx)
    validation_data_path = store.get_val_data_path(dataset_idx)

    if not store.exists(train_data_path):
        raise ValueError("{} path does not exist in the store".format(train_data_path))

    train_data = store.get_parquet_dataset(train_data_path)
    schema = train_data.schema.to_arrow_schema()
    train_rows, total_byte_size = _get_dataset_info(train_data, 'training', train_data_path)

    val_rows = 0
    if store.exists(validation_data_path):
        val_data = store.get_parquet_dataset(validation_data_path)
        val_rows, _ = _get_dataset_info(val_data, 'validation', validation_data_path)

    metadata = {}
    for col in schema_cols:
        col_schema = schema.field(col)
        col_info = {
            'spark_data_type': pyarrow_to_spark_data_type(col_schema.type),
            'is_sparse_vector_only': False,
            'shape': None,  # Only used by SparseVector columns
            'intermediate_format': constants.NOCHANGE,
            'max_size': None  # Only used by SparseVector columns
        }
        metadata[col] = col_info

    avg_row_size = total_byte_size / train_rows
    return train_rows, val_rows, metadata, avg_row_size


def _train_val_split(df, validation):
    train_df = df
    val_df = None
    validation_ratio = 0.0

    if isinstance(validation, float) and validation > 0:
        train_df, val_df = train_df.randomSplit([1.0 - validation, validation])
        validation_ratio = validation
    elif isinstance(validation, str):
        dtype = [field.dataType for field in df.schema.fields if field.name == validation][0]
        bool_dtype = isinstance(dtype, BooleanType)
        val_df = train_df.filter(
            f.col(validation) if bool_dtype else f.col(validation) > 0).drop(validation)
        train_df = train_df.filter(
            ~f.col(validation) if bool_dtype else f.col(validation) == 0).drop(validation)

        # Approximate ratio of validation data to training data for proportionate scale
        # of partitions
        timeout_ms = 1000
        confidence = 0.90
        train_rows = train_df.rdd.countApprox(timeout=timeout_ms, confidence=confidence)
        val_rows = val_df.rdd.countApprox(timeout=timeout_ms, confidence=confidence)
        validation_ratio = val_rows / (val_rows + train_rows)
    elif validation:
        raise ValueError('Unrecognized validation type: {}'.format(type(validation)))

    return train_df, val_df, validation_ratio


def _create_dataset(store, df, validation, compress_sparse,
                    num_partitions, num_workers, dataset_idx, parquet_row_group_size_mb, verbose):
    train_data_path = store.get_train_data_path(dataset_idx)
    val_data_path = store.get_val_data_path(dataset_idx)
    if verbose >= 1:
        print('CEREBRO => Time: {}, Writing DataFrames'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print('CEREBRO => Time: {}, Train Data Path: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                               train_data_path))
        print('CEREBRO => Time: {}, Val Data Path: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                             val_data_path))

    schema_cols = df.columns

    if isinstance(validation, str):
        schema_cols.append(validation)
    df = df[schema_cols]

    metadata = None
    if _has_vector_column(df):
        if compress_sparse:
            metadata = _get_metadata(df)
        to_petastorm = to_petastorm_fn(schema_cols, metadata)
        df = df.rdd.map(to_petastorm).toDF()

    train_df, val_df, validation_ratio = _train_val_split(df, validation)

    unischema_fields = []
    metadata = _get_metadata(train_df)
    for k in metadata.keys():
        type = spark_to_petastorm_type(metadata[k]['spark_data_type'])
        shape = petastorm_unischema_shape(metadata[k]['shape'])
        codec = petastorm_unischema_codec(metadata[k]['shape'], metadata[k]['spark_data_type'])
        unischema_fields.append(UnischemaField(k, type, shape, codec, False))

    petastorm_schema = Unischema('petastorm_schema', unischema_fields)

    train_partitions = max(int(num_partitions * (1.0 - validation_ratio)),
                           num_workers)
    if verbose >= 1:
        print('CEREBRO => Time: {}, Train Partitions: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                                train_partitions))

    spark = SparkSession.builder.getOrCreate()
    # FIXME pass hdfs_driver from user interface instead of hardcoded PETASTORM_HDFS_DRIVER
    train_resolver = FilesystemResolver(train_data_path,
                                        spark.sparkContext._jsc.hadoopConfiguration(),
                                        user=spark.sparkContext.sparkUser(),
                                        hdfs_driver=constants.PETASTORM_HDFS_DRIVER)
    with materialize_dataset(spark, train_data_path, petastorm_schema, parquet_row_group_size_mb,
                             filesystem_factory=train_resolver.filesystem_factory()):
        train_rdd = train_df.rdd.map(lambda x: x.asDict()).map(
            lambda x: {k: np.array(x[k], dtype=spark_to_petastorm_type(metadata[k]['spark_data_type'])) for k in x}) \
            .map(lambda x: dict_to_spark_row(petastorm_schema, x))

        spark.createDataFrame(train_rdd, petastorm_schema.as_spark_schema()) \
            .coalesce(train_partitions) \
            .write \
            .mode('overwrite') \
            .parquet(train_data_path)

    if val_df:
        val_partitions = max(int(num_partitions * validation_ratio),
                             num_workers)
        if verbose >= 1:
            print('CEREBRO => Time: {}, Val Partitions: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                                  val_partitions))
        val_resolver = FilesystemResolver(val_data_path,
                                          spark.sparkContext._jsc.hadoopConfiguration(),
                                          user=spark.sparkContext.sparkUser(),
                                          hdfs_driver=constants.PETASTORM_HDFS_DRIVER)
        with materialize_dataset(spark, val_data_path, petastorm_schema, parquet_row_group_size_mb,
                                 filesystem_factory=val_resolver.filesystem_factory()):
            val_rdd = val_df.rdd.map(lambda x: x.asDict()).map(
                lambda x: {k: np.array(x[k], dtype=spark_to_petastorm_type(metadata[k]['spark_data_type'])) for k in x}) \
                .map(lambda x: dict_to_spark_row(petastorm_schema, x))

            spark.createDataFrame(val_rdd, petastorm_schema.as_spark_schema()) \
                .coalesce(val_partitions) \
                .write \
                .mode('overwrite') \
                .parquet(val_data_path)

    train_rows, val_rows, pq_metadata, avg_row_size = get_simple_meta_from_parquet(store, train_df.columns, dataset_idx)

    if verbose:
        print(
        'CEREBRO => Time: {}, Train Rows: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), train_rows))
    if val_df:
        if val_rows == 0:
            raise ValueError(
                'Validation DataFrame does not any samples with validation param {}'
                    .format(validation))
        if verbose:
            print(
            'CEREBRO => Time: {}, Val Rows: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), val_rows))

    return train_rows, val_rows, pq_metadata, avg_row_size


def check_validation(validation, df=None):
    if validation:
        if isinstance(validation, float):
            if validation < 0 or validation >= 1:
                raise ValueError('Validation split {} must be in the range: [0, 1)'
                                 .format(validation))
        elif isinstance(validation, str):
            if df is not None and validation not in df.columns:
                raise ValueError('Validation column {} does not exist in the DataFrame'
                                 .format(validation))
        else:
            raise ValueError('Param validation must be of type "float" or "str", found: {}'
                             .format(type(validation)))


def prepare_data(num_workers, store, df,
                 validation=None, compress_sparse=False,
                 num_partitions=None, parquet_row_group_size_mb=8, dataset_idx=None, verbose=0):
    check_validation(validation, df=df)
    num_partitions = num_partitions or df.rdd.getNumPartitions()
    if num_workers <= 0 or num_partitions <= 0:
        raise ValueError('num_workers={} and partitions_per_process: {} must both be > 0'
                         .format(num_workers, num_partitions))

    if verbose >= 1:
        print('CEREBRO => Time: {}, Num Partitions: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                              num_partitions))

    return _create_dataset(store, df, validation, compress_sparse,
                           num_partitions, num_workers, dataset_idx, parquet_row_group_size_mb, verbose)


def to_list(var, length):
    if var is None:
        return None

    if not isinstance(var, list):
        var = [var]

    # If var has only one element, pad it to match the given length.
    if len(var) == 1:
        var = [var[0] for _ in range(length)]
    else:
        if len(var) != length:
            raise ValueError("loss_constructors and loss functions must be a "
                             "list with length that matches the length of "
                             "label_cols")

    return var
