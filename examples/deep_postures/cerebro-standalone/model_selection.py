# Copyright 2020 Supun Nakandala, Yuhao Zhang, and Arun Kumar. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import os
import sys

from cerebro.standalone import standalone_schedule as schedule
from datetime import datetime, timedelta
import h5py
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib
tf.disable_v2_behavior()
import numpy as np
import random

random.seed(2019)
tf.set_random_seed(2019)


##################################### 1. Input Function #######################################

def input_iterator(subject_file_path, train=True):
    fnames = [name.split('.')[0] for name in os.listdir(subject_file_path) if not name.startswith('.')]
    fnames.sort()

    for fname in fnames:
        h5f = h5py.File(os.path.join(subject_file_path,  '{}.h5'.format(fname)), 'r')
        timestamps = h5f.get('time')[:]
        data = h5f.get('data')[:]
        sleeping = h5f.get('sleeping')[:]
        non_wear = h5f.get('non_wear')[:]
        label = h5f.get('label')[:]

        data_batch = []
        timestamps_batch = []
        label_batch = []
        for d, t, s, nw, l in zip(data, timestamps, sleeping, non_wear, label):
            if (train and l == -1) or s == 1 or nw == 1:
                if len(timestamps_batch) > 0:
                    yield np.array(data_batch), np.array(timestamps_batch), np.array(label_batch)
                data_batch = []
                timestamps_batch = []
                label_batch = []
                continue

            data_batch.append(d)
            timestamps_batch.append(t)
            label_batch.append(l)
    
        if len(timestamps_batch) > 0:
            yield np.array(data_batch), np.array(timestamps_batch), np.array(label_batch)

        h5f.close()


def input_fn(file_path):
    x_segments = []; y_segments = []
    with open(file_path) as fin:
        for subject_file_path in fin:
            subject_file_path = subject_file_path.strip()
            for x, _, y in input_iterator(subject_file_path):
                x_segments.append(x)
                y_segments.append(y)
                

    temp = list(zip(x_segments, y_segments))
    random.shuffle(temp)
    x_segments, y_segments = zip(*temp)
                
    return x_segments, y_segments


################################## 2. Model Function ##########################################


def cnn_model(x, reuse, amp_factor=1):
    with tf.variable_scope('model', reuse=reuse):
        conv1 = tf.layers.conv2d(x, filters=32*amp_factor, kernel_size=[5, 3], data_format='channels_first', padding= "same",
                                 strides=(2, 1),
                                 activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(conv1, filters=64*amp_factor, kernel_size=[5, 1], data_format='channels_first', padding= "same",
                                 strides=(2, 1),
                                 activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(conv2, filters=128*amp_factor, kernel_size=[5, 1], data_format='channels_first', padding= "same",
                                 strides=(2, 1),
                                 activation=tf.nn.relu)

        conv4 = tf.layers.conv2d(conv3, filters=256*amp_factor, kernel_size=[5, 1], data_format='channels_first', padding= "same",
                                strides=(2, 1), 
                                activation=tf.nn.relu)
        
        conv5 = tf.layers.conv2d(conv4, filters=256*amp_factor, kernel_size=[5, 1], data_format='channels_first', padding= "same",
                                strides=(2, 1), 
                                activation=tf.nn.relu)
        size = conv5.shape[-1] * conv5.shape[-2] * conv5.shape[-3]

        logits = tf.layers.dense(tf.reshape(conv5,(-1, size)), units=256*amp_factor)

        return logits


def cnn_bi_lstm_model(x, reuse, amp_factor, win_size_mins):
    logits = cnn_model(x, reuse, amp_factor=amp_factor)
    logits = tf.reshape(logits, [-1, win_size_mins*6, 256*amp_factor])

    forward_cell = tf.nn.rnn_cell.LSTMCell(128)
    backward_cell = tf.nn.rnn_cell.LSTMCell(128)
    encoder_outputs,_ = tf.nn.bidirectional_dynamic_rnn(
            forward_cell,
            backward_cell,
            logits,
            dtype=tf.float32
        )
    encoder_outputs = tf.concat(encoder_outputs, axis=2)
    logits = tf.reshape(tf.layers.dense(encoder_outputs, units=1), [-1, win_size_mins*6])
    pred = tf.round(tf.nn.sigmoid(logits))
    return logits, pred



def window_generator(data, win_size_mins):
    win_size_10s = win_size_mins * 6

    for x_seg, y_seg in zip(*data):
        x_window = []; y_window = []
        for x,y in zip(x_seg, y_seg):
            x_window.append(x)
            y_window.append(y)

            if len(y_window) == win_size_10s:
                yield np.stack(x_window, axis=0), np.stack(y_window, axis=0)
                x_window = []; y_window = []


def model_fn(data, train_config):
    learning_rate = train_config['learning_rate']
    reg_value = train_config['reg_param']
    amp_factor = train_config['amp_factor']
    win_size_mins = train_config['win_size_mins']
    batch_size = train_config['batch_size']

    pre_processed_data_freq = 10 #Hz
    cnn_window_size = 10 # s
    num_cnn_data_points = pre_processed_data_freq * cnn_window_size
    win_size_cnn_outputs = win_size_mins * (60//cnn_window_size)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    dataset = tf.data.Dataset.from_generator(
        lambda: window_generator(data, win_size_mins),output_types=(tf.float32, tf.float32),
        output_shapes=((win_size_cnn_outputs, num_cnn_data_points, 3), (win_size_cnn_outputs))).prefetch(-1).batch(batch_size)
        
    data_iterator = dataset.make_one_shot_iterator()
    x, y = data_iterator.get_next()

    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
    num_gpus = min(1, len(gpu_names))
    
    # 10 seconds at 10 Hz
    x = tf.reshape(x,[-1, 1, num_cnn_data_points, 3])
    y = tf.reshape(y, [-1, win_size_cnn_outputs])
    x_vals = tf.split(x, num_gpus, axis=0)
    y_vals = tf.split(y, num_gpus, axis=0)
        
    losses = []; predictions = []; actual_labels = []
    for gpu_id in range(int(num_gpus)):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                logits, pred = cnn_bi_lstm_model(x_vals[gpu_id], (gpu_id > 0), amp_factor, win_size_mins)
                predictions.append(pred)
                l2 = reg_value * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
                actual_labels.append(y_vals[gpu_id])
                losses.append(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.cast(y_vals[gpu_id], tf.float32), logits=logits)) + l2)
    
        
    loss = tf.reduce_mean(losses)
    prediction = tf.concat(predictions, axis=0)
    label = tf.concat(actual_labels, axis=0)
    error = 1 - tf.reduce_mean(tf.cast(tf.equal(prediction, label), tf.float32))
    train_step = optimizer.minimize(loss, colocate_gradients_with_ops=True) 
    
    return optimizer, loss, error

###################################### 3. Train Function ########################################

def train_fn(sess, train_step, loss, error, train=True):
    losses = []
    errors = []
    while True:
        try:
            if train:
                _, train_l, train_e = sess.run([train_step, loss, error])
            else:
                train_l, train_e = sess.run([loss, error])
            losses.append(train_l)
            errors.append(train_e)
        except tf.errors.OutOfRangeError:
            break
        except Exception as e:
            print(e)
    return sum(losses) / len(losses), sum(errors) / len(errors)


##################################### 4. Eval Function ##########################################
def eval_fn(config_state):
    """
    :param config_state:
    """
    
    # Grid search: simply run all configurations for a fixed number of epochs (e.g., 10)
    stop_list = []
    for config_id in config_state:
        if len(config_state[config_id]['train_loss']) == 10:
            stop_list.append(config_id)

    new_configs = []
    return stop_list, new_configs



def main():
    # Define Cerebro ROOT directory.
    CEREBRO_ROOT = '/home/snakanda/Work/cerebro-system'
    
    # Define Cerebro Workers
    workers = ["http://0.0.0.0:7777"] # Add more workers if you have machines. You can start a wroker by running `cerebro-standalone-worker` command.

    # Pre-process the data using the utilities provided in the DeepPostures library: https://github.com/ADALabUCSD/DeepPostures/tree/master/MSSE-2021#pre-processing-data
    # Create partition definition files and store them in data/train and data/valid directories. Example files are provided.
    # Number of train paritions has to be same as the number of valid partitions.
    train_partitions = ['{}/examples/deep_postures/data/train/train_'.format(CEREBRO_ROOT) + str(i) + '.txt' for i in range(len(workers))]
    valid_paritions = ['{}/examples/deep_postures/data/valid/valid_'.format(CEREBRO_ROOT) + str(i) + '.txt' for i in range(len(workers))]
    
    # Create an availability matrix where each parition is available only on a single worker.
    # Data will be loaded into worker memory.
    avaliability = [[0 for _ in range(len(worker_ids))] for _ in range(len(worker_ids))]
    for i in range(len(worker_ids)):
        for j in range(len(worker_ids)):
            if i == j:
                avaliability[i][j] = 1

    # Parameter search space definition.
    param_grid = {
            'learning_rate': [1e-3, 1e-4],
            'reg_param': [1e-3, 1e-4],
            'amp_factor': [2, 4],
            'win_size_mins': [7, 9],
            'batch_size': [128]
    }

    param_names = [x for x in param_grid.keys()]

    def find_combinations(combinations, p, i):
        """
        :param combinations:
        :param p:
        :param i:
        """
        if i < len(param_names):
            for x in param_grid[param_names[i]]:
                p[param_names[i]] = x
                find_combinations(combinations, p, i + 1)
        else:
            combinations.append(p.copy())

    # Grid search. Creating all parameter configuration value sets.
    train_configs = []
    find_combinations(train_configs, {}, 0)

    #Creating model checkpoints directory.
    ckpt_root = '{}/examples/deep_postures/checkpoints'.format(CEREBRO_ROOT)
    if not os.path.exists(ckpt_root):
        os.makedirs(ckpt_root)

    # Invoking the model selection workload.
    schedule(workers, train_partitions, valid_paritions, avaliability, avaliability,
            input_fn=input_fn,
            model_fn=model_fn,
            train_fn=train_fn,
            initial_msts=train_configs,
            mst_eval_fn=eval_fn,
            ckpt_root=ckpt_root,
            preload_data_to_mem=True)


if __name__ == "__main__":
    main()