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

import io
import warnings

import h5py
import datetime
import numpy as np
import tensorflow as tf
from calendar import timegm
from decimal import Decimal

from .. import optimizer
from . import params
from ...backend import codec
from ...backend import constants

TF_KERAS = 'tf_keras'


class TFKerasUtil(object):
    type = TF_KERAS

    @staticmethod
    def fit_sub_epoch_fn(max_input_queue_size, input_queue_num_proc):
        def fn(starting_epoch, model, train_data, steps_per_epoch, callbacks, verbose):
            print('===> Before fit')
            return model.fit(
                train_data,
                initial_epoch=starting_epoch,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks,
                verbose=verbose,
                epochs=starting_epoch + 1,
                # use_multiprocessing=True,
                max_queue_size=max_input_queue_size,
                workers=input_queue_num_proc
            )

        return fn

    @staticmethod
    def eval_sub_epoch_fn(max_input_queue_size, input_queue_num_proc):
        def fn(_, model, val_data, validation_steps, callbacks, verbose):
            return model.evaluate(val_data, steps=validation_steps, callbacks=callbacks,
                                  use_multiprocessing=True,
                                  max_queue_size=max_input_queue_size,
                                  workers=input_queue_num_proc,
                                  verbose=verbose)

        return fn

    @staticmethod
    def make_dataset_fn(feature_columns, label_columns, sample_weight_col, metadata,
                        input_shapes, output_shapes, output_names, batch_size):
        # Check if any of the columns are only SparseVector
        has_sparse_col = any(metadata[col]['is_sparse_vector_only']
                             for col in label_columns + feature_columns)

        reshape = TFKerasUtil._reshape_fn(
            sample_weight_col, feature_columns, label_columns, metadata)
        prep_data_tf_keras = _prep_data_fn(
            has_sparse_col, sample_weight_col, feature_columns,
            label_columns, input_shapes, output_shapes, output_names)

        def fn(reader, shuffle_buffer_size, shuffle=False):
            # from petastorm.tf_utils import make_petastorm_dataset
            # dataset = make_petastorm_dataset(reader).unbatch()

            dataset = _make_petastorm_dataset(reader).unbatch()

            if shuffle:
                dataset = dataset.shuffle(shuffle_buffer_size)

            # Decompress sparse data if necessary
            if has_sparse_col:
                dataset = dataset.batch(1).map(reshape, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            dataset = dataset.batch(batch_size) \
                .map(prep_data_tf_keras, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            return dataset

        return fn

    @staticmethod
    def keras():
        return TFKerasUtil.keras_fn()()

    @staticmethod
    def keras_fn():
        def fn():
            import tensorflow.keras as tf_keras
            return tf_keras

        return fn

    @staticmethod
    def serialize_optimizer(*args, **kwargs):
        return optimizer.serialize_tf_keras_optimizer(*args, **kwargs)

    @staticmethod
    def deserialize_optimizer(*args, **kwargs):
        return optimizer.deserialize_tf_keras_optimizer(*args, **kwargs)

    @staticmethod
    def serialize_model(*args, **kwargs):
        def serialize_keras_model(x):
            return _serialize_keras_model(x, tf.keras.models.save_model)

        return serialize_keras_model(*args, **kwargs)

    @staticmethod
    def deserialize_model(*args, **kwargs):
        return _deserialize_keras_model(*args, **kwargs)

    @staticmethod
    def serialize_param_value(*args, **kwargs):
        def _serialize_param(x, y):
            return _serialize_param_value(x, y,
                                          serialize_model_fn=TFKerasUtil.serialize_model,
                                          serialize_opt_fn=TFKerasUtil.serialize_optimizer)

        return _serialize_param(*args, **kwargs)

    @staticmethod
    def _reshape_fn(sample_weight_col, feature_columns, label_columns, metadata):
        CUSTOM_SPARSE = constants.CUSTOM_SPARSE
        custom_sparse_to_dense = _custom_sparse_to_dense_fn()

        def reshape(row):
            new_row = {}
            if sample_weight_col:
                new_row[sample_weight_col] = getattr(row, sample_weight_col)

            for col in feature_columns + label_columns:
                v = getattr(row, col)
                intermediate_format = metadata[col]['intermediate_format']
                if intermediate_format == CUSTOM_SPARSE:
                    reshaped_v = tf.reshape(v, [metadata[col]['max_size'] * 2 + 1])
                    v = custom_sparse_to_dense(reshaped_v, metadata[col]['shape'])

                new_row[col] = v
            return new_row

        return reshape


########################################################################################################################


def _make_petastorm_dataset(reader):
    def dequeue_sample_impl():
        if reader.last_row_consumed:
            # This means that Dataset is trying to create a new instance of the generator. Can not do that
            # (nor want to do that) since this is an expensive operation. num_epochs is a more efficient way
            # to do this.
            raise RuntimeError('Multiple iterations over make_petastorm_dataset are not supported. '
                               'Multiple iterations can be triggered by calling \'repeat\' method of Datset class.'
                               'Use Reader\'s num_epochs contructor arguments to set number of iterations.')
        for row in reader:
            yield _sanitize_field_tf_types(row)

    flat_dataset = tf.data.Dataset.from_generator(dequeue_sample_impl, tuple(_schema_to_tf_dtypes(reader.schema)))

    # Don't write this function as a inline lambda like `dataset.map(lambda row: _set_shape_to_named_tuple(...))`,
    # It can avoid this error: https://github.com/tensorflow/tensorflow/issues/30149
    def set_shape(row):
        return _set_shape_to_named_tuple(reader.schema, row, reader.batched_output)

    schema_tuple = reader.schema._get_namedtuple()
    named_tuple_dataset = flat_dataset \
        .map(schema_tuple, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(set_shape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return named_tuple_dataset


def _set_shape_to_named_tuple(schema, fields, batched_output):
    """Assign static shape for all tensors"""
    fields_as_dict = fields._asdict()
    _set_shape(schema, fields_as_dict, batched_output)
    return schema.make_namedtuple_tf(**fields_as_dict)


def _set_shape(schema, fields_as_dict, batched_output=None):
    # Assign static shape for all tensors
    # Workaround of an issue described here:
    # https://stackoverflow.com/questions/49161316/trailing-x00-characters-in-tensor-when-numpy-string-array-is-returned-from-tf
    for k in fields_as_dict.keys():
        unischema_field = schema.fields[k]

        if fields_as_dict[k].get_shape().dims is None:
            if batched_output:
                shape = (None,) + unischema_field.shape
            else:
                shape = unischema_field.shape
            # Set static shape
            fields_as_dict[k].set_shape(shape)


def _schema_to_tf_dtypes(schema):
    """Returns schema as a list of tensorflow dtypes.
    :param schema: The schema.
    :return: List of tensorflow dtypes.
    """
    return [_numpy_to_tf_dtypes(f.numpy_dtype) for f in schema.fields.values()]


# Mapping of identical datatypes in numpy-ish and tensorflow-ish
_NUMPY_TO_TF_DTYPES_MAPPING = {
    np.bool: tf.bool,
    np.int8: tf.int8,
    np.int16: tf.int16,
    np.int32: tf.int32,
    np.int64: tf.int64,
    np.uint8: tf.uint8,
    np.uint16: tf.int32,
    np.uint32: tf.int64,
    np.float32: tf.float32,
    np.float64: tf.float64,
    np.string_: tf.string,
    np.unicode_: tf.string,
    np.str_: tf.string,
    np.bool_: tf.bool,
    Decimal: tf.string,
    np.datetime64: tf.int64,
}


def _numpy_to_tf_dtypes(numpy_dtype):
    """Returns a tensorflow dtype object corresponding to numpy's dtype.
    A :class:`ValueError` is raised if there is no known mapping between the types
    :param numpy_dtype: numpy dtype object
    :return: tensorflow dtype object
    """
    if numpy_dtype in _NUMPY_TO_TF_DTYPES_MAPPING:
        if numpy_dtype == np.unicode_ and sys.version_info >= (3, 0):
            warnings.warn("Tensorflow will convert all unicode strings back to bytes type. "
                          "You may need to decode values.", UnicodeWarning)
        return _NUMPY_TO_TF_DTYPES_MAPPING[numpy_dtype]
    else:
        raise ValueError('Unknown mapping of numpy {} to tensorflow dtype'.format(numpy_dtype))


def date_to_nsec_from_epoch(dt):
    return timegm(dt.timetuple()) * 1000000000


_date_to_nsec_from_epoch_vectorized = np.vectorize(date_to_nsec_from_epoch)


def _sanitize_field_tf_types(sample):
    """Takes a named tuple and casts/promotes types unknown to TF to the types that are known.
    Three casts that are currently implemented
      - Decimal to string
      - uint16 to int32
      - np.datetime64 to int64, as nanoseconds since unix epoch
    :param sample: named tuple or a dictionary
    :return: same type as the input with values casted to types supported by Tensorflow
    """
    next_sample_dict = sample._asdict()

    for k, v in next_sample_dict.items():
        if v is None:
            raise RuntimeError('Encountered "{}"=None. Tensorflow does not support None values as a tensor.'
                               'Consider filtering out these rows using a predicate.'.format(k))
        # Assuming conversion to the same numpy type is trivial and dirty cheap
        if isinstance(v, Decimal):
            # Normalizing decimals only to get rid of the trailing zeros (makes testing easier, assuming has
            # no other effect)
            next_sample_dict[k] = str(v.normalize())
        elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.datetime64):
            # Convert to nanoseconds from POSIX epoch
            next_sample_dict[k] = (v - np.datetime64('1970-01-01T00:00:00.0')) \
                .astype('timedelta64[ns]').astype(np.int64)
        elif isinstance(v, np.ndarray) and v.dtype == np.uint16:
            next_sample_dict[k] = v.astype(np.int32)
        elif isinstance(v, np.ndarray) and v.dtype == np.uint32:
            next_sample_dict[k] = v.astype(np.int64)
        elif isinstance(v, np.ndarray) and v.dtype.type in (np.bytes_, np.unicode_):
            if v.size != 0:
                next_sample_dict[k] = v.tolist()
        elif isinstance(v, np.ndarray) and v.dtype.kind == 'O' and isinstance(v[0], datetime.date):
            # Pyarrow 0.12.1 started returning python datetime.date when parquet column is a DateType() column.
            # Convert values in such column into nsec from epoch int64.
            next_sample_dict[k] = _date_to_nsec_from_epoch_vectorized(v)

    # Construct object of the same type as the input
    return sample.__class__(**next_sample_dict)


########################################################################################################################

def _prep_data_fn(has_sparse_col, sample_weight_col, feature_columns, label_columns,
                  input_shapes, output_shapes, output_names):
    def _get_from_dict(row, col):
        return row[col]

    def _get_from_named_tuple(row, col):
        return getattr(row, col)

    if has_sparse_col:
        get_col_from_row_fn = _get_from_dict
    else:
        get_col_from_row_fn = _get_from_named_tuple

    num_inputs = len(feature_columns)
    num_labels = len(label_columns)

    def prep(row):
        if sample_weight_col:
            sample_weight = get_col_from_row_fn(row, sample_weight_col)
            return (
                tuple(
                    tf.reshape(get_col_from_row_fn(row, feature_columns[i]), input_shapes[i])
                    for i
                    in range(num_inputs)),
                tuple(
                    tf.reshape(get_col_from_row_fn(row, label_columns[j]), output_shapes[j]) for
                    j
                    in range(num_labels)),
                {name: tf.reshape(sample_weight, [-1]) for name in output_names}
            )
        else:
            return (
                tuple(
                    tf.reshape(get_col_from_row_fn(row, feature_columns[i]), input_shapes[i])
                    for i
                    in range(num_inputs)),
                tuple(
                    tf.reshape(get_col_from_row_fn(row, label_columns[j]), output_shapes[j]) for
                    j
                    in range(num_labels))
            )

    return prep


def _serialize_keras_model(model, save_model_fn):
    """Serialize model into byte array encoded into base 64."""
    bio = io.BytesIO()
    with h5py.File(bio, 'w') as f:
        save_model_fn(model, f)
    return codec.dumps_base64(bio.getvalue())


def _deserialize_keras_model(model_bytes, load_model_fn):
    model_bytes = codec.loads_base64(model_bytes)
    bio = io.BytesIO(model_bytes)
    with h5py.File(bio, 'r') as f:
        return load_model_fn(f)


def _serialize_param_value(param_name, param_val, serialize_model_fn, serialize_opt_fn):
    if param_val is None:
        return param_val

    if param_name in [params.SparkEstimatorParams.backend.name, params.SparkEstimatorParams.store.name]:
        # We do not serialize backend and store. These params have to be regenerated for each
        # run of the pipeline
        return None
    elif param_name == params.SparkEstimatorParams.model.name:
        return serialize_model_fn(param_val)
    if param_name == params.SparkEstimatorParams.optimizer.name:
        return serialize_opt_fn(param_val)
    else:
        return codec.dumps_base64(param_val)


def _custom_sparse_to_dense_fn():
    # TODO(fardin): ask petastorm team about codecs for sparse and dense vectors and see if that is
    # a better solution
    def custom_sparse_to_dense(custom_sparse_vec, dense_shape):
        # original sparse vector:   v = {1:2.0, 3:.4.5, 5:7.1}
        # custom sparse vector:     v = [3, 1, 3, 5, 2.0, 4.5, 7.1]
        # dense vector:             v = [0, 2.0, 0, 4.5, 0, 7.1]

        # Get the first element from custom_sparse_vec. This element is the size of
        # non-zero elements in the original sparse vector.
        sparse_vector_size = tf.cast(tf.gather(custom_sparse_vec, 0, axis=0), tf.int32)
        sparse_vector_size = tf.reshape(sparse_vector_size, [1])

        # get the first sparse_vector_size elements of the custom_sparse_vec which are the
        # indices
        indices_1d = tf.cast(
            tf.slice(custom_sparse_vec, begin=tf.constant([1]), size=sparse_vector_size),
            tf.int64)
        indices_reshaped = tf.reshape(indices_1d,
                                      tf.concat([sparse_vector_size, tf.constant([1])], 0))
        # have to pad the indices to match the expected format by the SparseTensor
        indices = tf.pad(indices_reshaped, [[0, 0], [1, 0]], "CONSTANT")

        # get the second sparse_vector_size elements of the custom_sparse_vec which are
        # the values
        begin_index = sparse_vector_size + tf.constant(1)
        values = tf.slice(custom_sparse_vec, begin=begin_index, size=sparse_vector_size)

        # construct a sparse vector with the indices and values
        dense_shape = [1, dense_shape]
        sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values,
                                               dense_shape=dense_shape)
        # convert the sparse vector into a dense vector
        return tf.sparse.to_dense(sparse_tensor)

    return custom_sparse_to_dense
