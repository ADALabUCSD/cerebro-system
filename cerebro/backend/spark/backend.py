# Copyright 2020 University of California Regents. All Rights Reserved.
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
import math
import os
import random
import threading
import time
from distutils.version import LooseVersion

import h5py
import numpy as np
import pyspark
import tensorflow as tf
from ..backend import Backend
from six.moves import queue

from .. import timeout, codec, settings as spark_settings, secret, host_hash, job_id

from .. import constants
from . import service_driver, service_task, util

PETASTORM_HDFS_DRIVER = constants.PETASTORM_HDFS_DRIVER
TOTAL_BUFFER_MEMORY_CAP_GIB = constants.TOTAL_BUFFER_MEMORY_CAP_GIB
BYTES_PER_GIB = constants.BYTES_PER_GIB


def default_num_workers():
    spark_context = pyspark.SparkContext._active_spark_context
    return spark_context.defaultParallelism


class SparkBackend(Backend):
    """Uses `horovod.spark.run` to execute the distributed training `fn`."""

    def __init__(self, spark_context=None, num_workers=None, start_timeout=600, disk_cache_size=20480, verbose=1):
        """
        Args:
            spark_context: Spark context
            num_workers: Number of Cerebro workers.  Defaults to `spark.default.parallelism`.
            start_timeout: Timeout for Spark tasks to spawn, register and start running the code, in seconds.
                       If not set, falls back to `CEREBRO_SPARK_START_TIMEOUT` environment variable value.
                       If it is not set as well, defaults to 600 seconds.
            disk_cache_size: Size of the disk data cache in MBs.
            verbose: Debug output verbosity (0-2). Defaults to 1..
        """

        tmout = timeout.Timeout(start_timeout,
                                message='Timed out waiting for {activity}. Please check that you have '
                                        'enough resources to run all Horovod processes. Each Horovod '
                                        'process runs in a Spark task. You may need to increase the '
                                        'start_timeout parameter to a larger value if your Spark resources '
                                        'are allocated on-demand.')
        settings = spark_settings.Settings(verbose=verbose,
                                           key=secret.make_secret_key(),
                                           timeout=tmout,
                                           run_func_mode=True)

        self.disk_cache_size = disk_cache_size

        if spark_context is None:
            spark_context = pyspark.SparkContext._active_spark_context
            if spark_context is None:
                raise Exception('Could not find an active SparkContext, are you '
                                'running in a PySpark session?')
        self.spark_context = spark_context

        if num_workers is None:
            num_workers = spark_context.defaultParallelism
            if settings.verbose >= 1:
                print('Running %d processes (inferred from spark.default.parallelism)...' % num_workers)
        else:
            if settings.verbose >= 1:
                print('Running %d processes...' % num_workers)

        settings.num_workers = num_workers
        self.settings = settings

        self.workers_initialized = False
        self.task_clients = None
        self.driver = None
        self.driver_client = None
        self.spark_job_group = None
        self.data_loaders_initialized = False

    def initialize_workers(self):
        """Initialize Spark tasks"""
        result_queue = queue.Queue(1)
        spark_job_group = 'cerebro.spark.run.%d' % job_id.next_job_id()
        driver = service_driver.SparkDriverService(self.settings.num_workers, self.settings.key, self.settings.nics)
        driver_client = service_driver.SparkDriverClient(driver.addresses(), self.settings.key, self.settings.verbose)

        _make_spark_thread(self.spark_context, spark_job_group, driver, result_queue, self.settings)

        driver.wait_for_initial_registration(self.settings.timeout)
        if self.settings.verbose >= 2:
            print('Initial Spark task registration is complete.')
        task_clients = [service_task.SparkTaskClient(index,
                                                     driver.task_addresses_for_driver(index),
                                                     self.settings.key, self.settings.verbose) for index in
                        range(self.settings.num_workers)]
        for task_client in task_clients:
            task_client.notify_initial_registration_complete()

        # setting local index for each task on the corresponding worker for GPU pinning (if needed)
        host_hashes = driver.task_host_hash_indices()
        for host_hash in host_hashes:
            for i, task_index in enumerate(host_hashes[host_hash]):
                task_clients[task_index].set_local_task_index(i)

        self.driver = driver
        self.driver_client = driver_client
        self.task_clients = task_clients
        self.spark_job_group = spark_job_group
        self.workers_initialized = True

    def initialize_data_loaders(self, store, dataset_idx, schema_fields):
        """

        :param store:
        :param dataset_idx:
        :param schema_fields:
        """
        if self.workers_initialized:
            remote_store = store.to_remote(self.spark_job_group, dataset_idx)
            shard_count = self.num_workers()
            _, _, _, avg_row_size = util.get_simple_meta_from_parquet(store, schema_fields, None, dataset_idx)
            data_readers_fn = _data_readers_fn(remote_store, shard_count, schema_fields, avg_row_size,
                                               self.disk_cache_size)

            for task_client in self.task_clients:
                task_client.initialize_data_loaders(data_readers_fn)

            self.data_loaders_initialized = False
        else:
            raise Exception('Spark tasks not initialized for Cerebro. Please run SparkBackend.initialize_workers() '
                            'first!')

    def train_for_one_epoch(self, models, store, dataset_idx, feature_col, label_col, is_train=True):
        sub_epoch_trainers = [_get_remote_trainer(model, self, store, dataset_idx, feature_col, label_col, self.settings.verbose) \
                              for model in models]

        model_worker_pairs = [(i, j) for i in range(len(models)) for j in range(self.num_workers())]
        # take a random ordering
        random.shuffle(model_worker_pairs)

        model_states = {i: False for i in range(len(models))}
        worker_states = {i: False for i in range(self.num_workers())}
        model_on_worker = [-1 for _ in range(self.num_workers())]

        model_results = {model.getRunId(): None for model in models}

        while len(model_worker_pairs) > 0:

            for w in range(self.num_workers()):
                # worker idle
                if not worker_states[w]:
                    m = _get_runnable_model(w, model_worker_pairs, model_states)
                    if m != -1:
                        # runnable model found
                        self.task_clients[w].execute_sub_epoch(
                            fn=sub_epoch_trainers[m], train=is_train, initial_epoch=models[m].getEpochs())

                        model_states[m] = True
                        worker_states[w] = True
                        model_on_worker[w] = m
                else:
                    m = model_on_worker[w]
                    if m != -1:
                        status = self.task_clients[w].sub_epoch_completed()
                        if status.flag:
                            # sub-epoch completed
                            model_worker_pairs.remove((m, w))
                            model_states[m] = False
                            worker_states[w] = False
                            model_on_worker[w] = -1

                            res = status.sub_epoch_result['result']
                            run_id = models[m].getRunId()
                            if model_results[run_id] is None:
                                model_results[run_id] = res
                            else:
                                for k in model_results[run_id]:
                                    model_results[run_id][k].append(res[k][0])

            time.sleep(self.settings.polling_period)

        # incrementing the model epoch number
        if is_train:
            for model in models:
                model.setEpochs(model.getEpochs() + 1)

        # aggregating the model metrics
        for run_id in model_results:
            res = model_results[run_id]
            for k in res:
                res[k] = np.mean(res[k])

        return model_results

    def teardown_workers(self):
        """Teardown Spark tasks"""
        for task_client in self.task_clients:
            task_client.notify_workload_complete()

        self.workers_initialized = False
        self.data_loaders_initialized = False

    def get_metadata_from_parquet(self, store, label_columns=['label'], feature_columns=['features']):
        """
        Get metadata from the data in the persistent storage.
        :param store:
        :param label_columns:
        :param feature_columns:
        :return:
        """
        return util.get_simple_meta_from_parquet(store, label_columns + feature_columns)

    def prepare_data(self, store, dataset, validation, label_columns=['label'], feature_columns=['features'],
                     compress_sparse=False, verbose=2, dataset_idx=None):
        """
        Prepare data by writing out into persistent storage
        :param store:
        :param dataset:
        :param validation:
        :param label_columns:
        :param feature_columns:
        :param compress_sparse:
        :param verbose:
        :param dataset_idx:
        """
        return util.prepare_data(self.num_workers(), store, dataset, label_columns, feature_columns, validation,
                                 partitions_per_process=1, compress_sparse=compress_sparse, verbose=verbose,
                                 dataset_idx=dataset_idx)

    def num_workers(self):
        """
            Get number of processes/tasks
        :return:
        """
        return self.settings.num_workers


def _get_runnable_model(worker, model_worker_pairs, model_states):
    for m, w in model_worker_pairs:
        # worker matches and model idle
        if w == worker and not model_states[m]:
            return m
    return -1


def _get_remote_trainer(estimator, backend, store, dataset_idx, feature_columns, label_columns, verbose=0):
    train_rows, val_rows, metadata, avg_row_size = \
        util.get_simple_meta_from_parquet(store,
                                          schema_cols=label_columns + feature_columns,
                                          sample_weight_col=None,
                                          dataset_idx=dataset_idx)
    estimator._check_params(metadata)
    keras_utils = estimator._get_keras_utils()
    run_id = estimator.getRunId()
    if estimator._has_checkpoint(run_id):
        serialized_model = estimator._load_model_from_checkpoint(run_id)
    else:
        serialized_model = estimator._compile_model(keras_utils)

    trainer = sub_epoch_trainer(estimator, metadata, keras_utils, run_id, serialized_model, dataset_idx,
                                train_rows, val_rows, backend.num_workers(), verbose)
    return trainer


def _data_readers_fn(remote_store, shard_count, schema_fields, avg_row_size, cache_size_limit):
    def _data_readers(index):
        from petastorm import make_batch_reader

        PETASTORM_HDFS_DRIVER = constants.PETASTORM_HDFS_DRIVER

        train_reader = make_batch_reader(remote_store.train_data_path, shuffle_row_groups=False, num_epochs=None,
                                         cur_shard=index,
                                         shard_count=shard_count,
                                         hdfs_driver=PETASTORM_HDFS_DRIVER,
                                         schema_fields=schema_fields,
                                         cache_type='local-disk',
                                         cache_size_limit=cache_size_limit,
                                         cache_row_size_estimate=avg_row_size)

        if remote_store.val_data_path != '' and remote_store.val_data_path is not None:
            val_reader = make_batch_reader(remote_store.val_data_path, shuffle_row_groups=False, num_epochs=None,
                                           cur_shard=index,
                                           shard_count=shard_count,
                                           hdfs_driver=PETASTORM_HDFS_DRIVER,
                                           schema_fields=schema_fields,
                                           cache_type='local-disk',
                                           cache_size_limit=cache_size_limit,
                                           cache_row_size_estimate=avg_row_size)
        else:
            val_reader = None

        return train_reader, val_reader

    return _data_readers


def _make_spark_thread(spark_context, spark_job_group, driver, result_queue,
                       settings):
    """Creates `settings.num_workers` Spark tasks in a parallel thread."""

    def run_spark():
        """Creates `settings.num_workers` Spark tasks, each executing `_task_fn` and waits for them to terminate."""
        try:
            spark_context.setJobGroup(spark_job_group,
                                      "Cerebro Spark Run",
                                      interruptOnCancel=True)
            procs = spark_context.range(0, numSlices=settings.num_workers)
            # We assume that folks caring about security will enable Spark RPC
            # encryption, thus ensuring that key that is passed here remains
            # secret.
            result = procs.mapPartitionsWithIndex(_make_mapper(driver.addresses(), settings)).collect()
            result_queue.put(result)
        except:
            driver.notify_spark_job_failed()
            raise

    spark_thread = threading.Thread(target=run_spark)
    spark_thread.start()
    return spark_thread


def _make_mapper(driver_addresses, settings):
    def _mapper(index, _):
        try:
            # https://www.google.com/search?q=keras+model+save+resource+temporarily+unavailable&oq=keras\
            # +mode&aqs=chrome.0.69i59l2j69i57j69i59j69i60l3j69i65.3390j0j4&sourceid=chrome&ie=UTF-8
            import os
            os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

            task = service_task.SparkTaskService(index, settings.key, settings.nics)
            driver_client = service_driver.SparkDriverClient(driver_addresses, settings.key, settings.verbose)

            driver_client.register_task(index, task.addresses(), host_hash.host_hash())
            task.wait_for_initial_registration(settings.timeout)

            task.wait_for_workload_completion()
            yield 0
        finally:
            task.shutdown()
            # yield _task_fn(index, driver_addresses, settings)

    return _mapper


def sub_epoch_trainer(estimator, metadata, keras_utils, run_id, serialized_model, dataset_idx, train_rows, val_rows,
                      num_workers, verbose=0):
    # Estimator parameters
    label_columns = estimator.getLabelCols()
    feature_columns = estimator.getFeatureCols()
    user_callbacks = estimator.getCallbacks()
    batch_size = estimator.getBatchSize()
    sample_weight_col = estimator.getSampleWeightCol()
    custom_objects = estimator.getCustomObjects()
    user_shuffle_buffer_size = estimator.getShufflingBufferSize()
    metrics_names = [name.__name__ if callable(name) else name for name in  estimator.getMetrics()]
    model_logs_dir = estimator.getLogsDir()
    user_verbose = estimator.getVerbose()

    # Model parameters
    input_shapes, output_shapes = estimator.get_model_shapes()
    output_names = estimator.getModel().output_names

    floatx = tf.keras.backend.floatx()
    make_dataset = keras_utils.make_dataset_fn(
        feature_columns, label_columns, sample_weight_col, metadata,
        input_shapes, output_shapes, output_names, batch_size)
    fit_sub_epoch_fn = keras_utils.fit_sub_epoch_fn()
    eval_sub_epoch_fn = keras_utils.eval_sub_epoch_fn()
    transformation_fn = estimator.getTransformationFn()
    transformation = transformation_fn if transformation_fn else None

    # Utility functions
    deserialize_keras_model = _deserialize_keras_model_fn()
    calculate_shuffle_buffer_size = _calculate_shuffle_buffer_size_fn()
    pin_gpu = _pin_gpu_fn()

    # Storage
    store = estimator.getStore()
    remote_store = store.to_remote(run_id, dataset_idx)

    def train(data_reader, is_train, starting_epoch, local_task_index=0):
        begin_time = time.time()
        tf.keras.backend.set_floatx(floatx)
        pin_gpu(local_task_index)

        # FIXME: Enable sub-epoch data shuffling
        # if not user_shuffle_buffer_size:
        #     shuffle_buffer_size = calculate_shuffle_buffer_size(
        #         hvd, avg_row_size, train_rows / num_workers)
        # else:
        #     shuffle_buffer_size = user_shuffle_buffer_size

        if not user_shuffle_buffer_size:
            shuffle_buffer_size = 1024 * 3
        else:
            shuffle_buffer_size = user_shuffle_buffer_size

        with tf.keras.utils.custom_object_scope(custom_objects):
            model = deserialize_keras_model(
                serialized_model, lambda x: tf.keras.models.load_model(x))

        # # Verbose mode 1 will print a progress bar
        verbose = user_verbose

        with remote_store.get_local_output_dir() as run_output_dir:
            callbacks = user_callbacks
            ckpt_file = os.path.join(run_output_dir, remote_store.checkpoint_filename)
            # restore model from checkpoint if it exists
            if os.path.exists(ckpt_file):
                model.load_weights(ckpt_file)

            steps_per_epoch = int(math.ceil(train_rows / batch_size / num_workers))

            # math.ceil because if val_rows is smaller than batch_size we still get the at least
            # one step. float(val_rows) because val_rows/batch_size evaluates to zero before
            # math.ceil
            validation_steps = int(math.ceil(float(val_rows) / batch_size / num_workers))

            schema_fields = feature_columns + label_columns
            if sample_weight_col:
                schema_fields.append(sample_weight_col)

            if is_train:
                train_data = make_dataset(data_reader, shuffle_buffer_size, shuffle=False)
                initialization_time = time.time()
                begin_time = time.time()
                result = fit_sub_epoch_fn(starting_epoch, model, train_data, steps_per_epoch, callbacks,
                                          verbose).history
                training_time = time.time() - begin_time
                begin_time = time.time()
                result = {'train_' + name: result[name] for name in result}
                model.save(ckpt_file)
            else:
                val_data = make_dataset(data_reader, shuffle_buffer_size, shuffle=False)
                initialization_time = time.time()
                begin_time = time.time()
                result = eval_sub_epoch_fn(starting_epoch, model, val_data, validation_steps, callbacks, verbose)
                training_time = time.time() - begin_time
                begin_time = time.time()
                result = [[x] for x in result]
                result = {k: v for k, v in zip(['val_loss'] + ['val_' + name for name in metrics_names], result)}

            tf.keras.backend.clear_session()

            if remote_store.saving_runs:
                remote_store.sync(run_output_dir)
            finalization_time = time.time() - begin_time

            if verbose >= 1:
                print('Model {} initialization time: {}, training time: {}, finalization time: {}'.format(
                    'train' if is_train else 'valid',
                    initialization_time, training_time, finalization_time))

            return result

    return train


def _deserialize_keras_model_fn():
    def deserialize_keras_model(model_bytes, load_model_fn):
        """Deserialize model from byte array encoded in base 64."""
        model_bytes = codec.loads_base64(model_bytes)
        bio = io.BytesIO(model_bytes)
        with h5py.File(bio, 'r') as f:
            return load_model_fn(f)

    return deserialize_keras_model


def _calculate_shuffle_buffer_size_fn():
    def calculate_shuffle_buffer_size(hvd, avg_row_size, train_row_count_per_worker):
        """
        Determines the shuffling buffer size such that each worker gets at most 1GB for shuffling
        buffer such that on a single machine, among all the workers on that machine, at most
        memory_cap_gb GB are allocated for shuffling buffer. Also, it ensures that the buffer size
        is identical among all the workers.

        example 1:
        memory_cap_gb = 4
        machine1: 8 workers
        machine2: 3 workers
        shuffle_buffer_size = 0.5 GB

        example 2:
        memory_cap_gb = 4
            machine1: 2 workers
            machine2: 3 workers
        shuffle_buffer_size = 1 GB

        example 3:
        memory_cap_gb = 4
            machine1: 2 workers
            machine2: 8 workers
            machine3: 5 workers
        shuffle_buffer_size = 0.5 GB
        """
        local_size = hvd.local_size()
        local_sizes = hvd.allgather([local_size])
        max_local_size = max(local_sizes)

        if max_local_size > TOTAL_BUFFER_MEMORY_CAP_GIB:
            shuffle_buffer_size = TOTAL_BUFFER_MEMORY_CAP_GIB * BYTES_PER_GIB / avg_row_size / max_local_size
        else:
            shuffle_buffer_size = BYTES_PER_GIB / avg_row_size

        return int(min(shuffle_buffer_size, train_row_count_per_worker))

    return calculate_shuffle_buffer_size


def _pin_gpu_fn():
    def fn(local_task_index):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[local_task_index], 'GPU')

    return fn


def _pin_cpu_fn():
    def fn():
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    return fn
