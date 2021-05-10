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

import h5py
import tensorflow as tf

from . import params
from .. import optimizer
from ...backend import codec
from ...backend import constants

TF_KERAS = 'tf_keras'


class TFKerasUtil(object):
    type = TF_KERAS

    @staticmethod
    def fit_sub_epoch_fn():
        def fn(starting_epoch, model, train_data, callbacks, verbose):
            return model.fit(
                train_data,
                initial_epoch=starting_epoch,
                callbacks=callbacks,
                verbose=verbose,
                epochs=starting_epoch + 1)

        return fn

    @staticmethod
    def eval_sub_epoch_fn():
        def fn(_, model, val_data, callbacks, verbose):
            return model.evaluate(val_data, callbacks=callbacks, verbose=verbose)

        return fn

    @staticmethod
    def make_dataset_fn(feature_columns, label_columns, metadata,
                        input_shapes, output_shapes, input_names, output_names, batch_size):
        # Check if any of the columns are only SparseVector
        has_sparse_col = any(metadata[col]['is_sparse_vector_only'] for col in label_columns + feature_columns)

        reshape = TFKerasUtil._reshape_fn(feature_columns, label_columns, metadata)
        prep_data_tf_keras = _prep_data_fn(has_sparse_col, input_names, label_columns, input_shapes, output_shapes, output_names)

        def fn(reader, transformation_fn):
            from petastorm.tf_utils import make_petastorm_dataset
            dataset = make_petastorm_dataset(reader)

            # Decompress sparse data if necessary
            if has_sparse_col:
                dataset = dataset.batch(1).map(reshape, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            if transformation_fn:
                # user provided custom transformation function
                dataset = transformation_fn(dataset)

            dataset = dataset.batch(batch_size).map(prep_data_tf_keras, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            return dataset.prefetch(tf.data.experimental.AUTOTUNE)

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
    def _reshape_fn(feature_columns, label_columns, metadata):
        CUSTOM_SPARSE = constants.CUSTOM_SPARSE
        custom_sparse_to_dense = _custom_sparse_to_dense_fn()

        def reshape(row):
            new_row = {}
            for col in feature_columns + label_columns:
                v = getattr(row, col)
                intermediate_format = metadata[col]['intermediate_format']
                if intermediate_format == CUSTOM_SPARSE:
                    reshaped_v = tf.reshape(v, [metadata[col]['max_size'] * 2 + 1])
                    v = custom_sparse_to_dense(reshaped_v, metadata[col]['shape'])

                new_row[col] = v
            return new_row

        return reshape


def _prep_data_fn(has_sparse_col, input_names, label_columns,
                  input_shapes, output_shapes, output_names):

    def get_col_from_row_fn(row, col):
        if type(row) == dict:
            return row[col]
        else:
            return getattr(row, col)

    num_inputs = len(input_names)
    num_labels = len(label_columns)

    def prep(row):
        return (
            tuple(
                tf.reshape(get_col_from_row_fn(row, input_names[i]), input_shapes[i])
                for i
                in range(num_inputs)),
            # No reshaping for the outputs.
            tuple(get_col_from_row_fn(row, label_columns[j]) for j in range(num_labels))
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
