# Copyright 2021 Supun Nakandala. All Rights Reserved.
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

import base64
import os
import random
import shutil
import time
import datetime
import xmlrpc.client as xc
from distutils.dir_util import copy_tree
import dill
import numpy as np
import sys
import pickle
from .utils import tf_execute_helper, uuid, mst_identifier, preload_data_helper

dill.settings["recurse"] = True


def preload_data(workers, input_fn_string, preload_fn_string, train_partitions, valid_partitions,
                 train_availability, valid_availability, scheduler_log_file, begin_time):
    """

    :param workers:
    :param input_fn_string:
    :param preload_fn_string:
    :param train_partitions:
    :param valid_partitions:
    :param train_availability:
    :param valid_availability:
    :param scheduler_log_file:
    :param begin_time:
    """
    for i, worker in workers.items():
        worker.initialize_worker()

    exec_ids = []
    for worker_id, worker in workers.items():

        data_partitions = []
        for availability, partitions in zip([train_availability, valid_availability],
                                            [train_partitions, valid_partitions]):
            for i, available in enumerate(availability[worker_id]):
                if available:
                    data_partitions.append((partitions[i]))

        exec_id = uuid()
        params = [input_fn_string, data_partitions]
        
        result = worker.execute(exec_id, preload_fn_string, params)
        status = dill.loads(base64.b64decode(result.data))

        if status != "LAUNCHED":
            raise Exception("Remote job launch failed. Reason: " + status)

        exec_ids.append((exec_id, worker_id))

    # wait for everything to finish
    while len(exec_ids) > 0:
        for exec_id, worker_id in exec_ids:
            worker = workers[worker_id]
            status = dill.loads(base64.b64decode(worker.status(exec_id).data))

            if status["status"] == "FAILED":
                print(status)
                raise Exception("Remote job execution failed")
            elif status["status"] == "INVALID ID":
                raise Exception("Invalid Id")
            elif status["status"] == "COMPLETED":
                exec_ids.remove((exec_id, worker_id))
                message = "TIME: %d, EVENT: PRELOAD_COMPLETED, WORKER: %d\n" % (time.time() - begin_time, worker_id)
                scheduler_log_file.write(message)
                print(message[:-1])
                scheduler_log_file.flush()
        time.sleep(1)


def launch_job(worker, epoch, partitions, ckpt_path, data_partition_names,
               input_fn_string, model_fn_string, train_fn_string, exec_fn_string, mst, train):
    """

    :param worker:
    :param epoch:
    :param partitions:
    :param ckpt_path:
    :param data_partition_names:
    :param input_fn_string:
    :param model_fn_string:
    :param train_fn_string:
    :param exec_fn_string:
    :param mst:
    :param train:
    :return:
    """
    exec_id = uuid()
    params = [epoch, partitions, ckpt_path,
              [data_partition_name for data_partition_name in data_partition_names],
              input_fn_string, model_fn_string, train_fn_string, mst, train]

    result = worker.execute(exec_id, exec_fn_string, params)
    status = dill.loads(base64.b64decode(result.data))
    if status != "LAUNCHED":
        raise Exception("Remote job launch failed. Reason: " + status)

    return exec_id


def check_finished(worker, exec_id):
    """

    :param worker:
    :param exec_id:
    :return:
    """
    result = worker.status(exec_id)
    status = dill.loads(base64.b64decode(result.data))

    if status["status"] == "FAILED":
        raise Exception("Remote job execution failed")
    elif status["status"] == "INVALID ID":
        raise Exception("Invalid Id")
    elif status["status"] == "COMPLETED":
        return True, status
    else:
        return False, status


def update_mst_evaluation_state(epoch_mst_evaluation_state, mst_evaluation_state):
    """

    :param epoch_mst_evaluation_state:
    :param mst_evaluation_state:
    :return:
    """
    for mode in ["train", "valid"]:
        for mst_id in epoch_mst_evaluation_state:
            mst_evaluation_state[mst_id][mode + "_loss"].append(sum(epoch_mst_evaluation_state[mst_id][mode + "_loss"]) / len(epoch_mst_evaluation_state[mst_id][mode + "_loss"]))
            mst_evaluation_state[mst_id][mode + "_error"].append(sum(epoch_mst_evaluation_state[mst_id][mode + "_error"]) / len(epoch_mst_evaluation_state[mst_id][mode + "_error"]))
            if mode == "train":
                mst_evaluation_state[mst_id]['epoch'] += 1

    return mst_evaluation_state


def evaluate_msts(mst_eval_fn, mst_evaluation_state, current_msts, ckpt_root):
    """

    :param mst_eval_fn:
    :param mst_evaluation_state:
    :param current_msts:
    :param ckpt_root:
    :return:
    """
    stop_mst_ids, new_msts = mst_eval_fn(mst_evaluation_state)
    for mst_id in stop_mst_ids:
        mst_evaluation_state[mst_id]["state"] = "COMPLETED"

    current_msts = [(mst_id, mst) for mst_id, mst in current_msts if mst_id not in stop_mst_ids]

    id_max = max(mst_evaluation_state.keys())
    for mst_id, new_mst in zip(range(id_max + 1, id_max + 1 + len(new_msts)), new_msts):

        ckpt_path = ckpt_root + "/" + str(mst_id) + "_" + uuid()
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        ckpt_path = ckpt_path + "/model"

        if 'init_ckpt_path' in new_mst:
            copy_tree(new_mst['init_ckpt_path'], ckpt_path)

        mst_evaluation_state[mst_id] = {"state": "RUNNING", "epoch": -1, "train_error": [], "train_loss": [],
                                        "valid_error": [], "valid_loss": [],
                                        "ckpt_path": ckpt_path,
                                        "mst": new_mst}
        log_file = open("./logs/" + str(mst_id) + ".log", 'a')
        log_message(log_file, "Checkpoint Path: " + ckpt_path + "\n")
        log_message(log_file, "MST: " + mst_identifier(new_mst) + "\n")
        if 'init_mst' in new_mst:
            log_message(log_file, "Init MST ID: " + str(new_mst['init_mst_id']) + "\n")
            log_message(log_file, "Init MST: " + mst_identifier(new_mst['init_mst']) + "\n")

            mst_evaluation_state[mst_id]['valid_error'] = [x for x in mst_evaluation_state[new_mst['init_mst_id']]['valid_error']]
            mst_evaluation_state[mst_id]['train_error'] = [x for x in mst_evaluation_state[new_mst['init_mst_id']]['train_error']]
            mst_evaluation_state[mst_id]['valid_loss'] = [x for x in mst_evaluation_state[new_mst['init_mst_id']]['valid_loss']]
            mst_evaluation_state[mst_id]['train_loss'] = [x for x in mst_evaluation_state[new_mst['init_mst_id']]['train_loss']]

            mst_evaluation_state[mst_id]['epoch'] = mst_evaluation_state[new_mst['init_mst_id']]['epoch']

        current_msts.append((mst_id, new_mst))

    return current_msts, mst_evaluation_state


def log_message(log_file, message, print_message=False):
    """

    :param log_file:
    :param message:
    :param print_message:
    """
    log_file.write(message)
    log_file.flush()
    os.fsync(log_file.fileno())
    if print_message:
        print(message[:-1])


def schedule(worker_ips, train_partitions, valid_partitions, train_availability, valid_availability,
             input_fn, model_fn, train_fn, initial_msts, mst_eval_fn,
                    ckpt_root='/tmp', preload_data_to_mem=True, backend='tf'):
    """
    :param workers:
    :param train_partitions:
    :param valid_partitions:    
    :param train_availability:
    :param valid_availability:    
    :param input_fn:
    :param model_fn:
    :param train_fn:
    :param initial_msts:
    :param mst_eval_fn:
    :param ckpt_root:
    :param preload_data_to_mem:
    """
    begin_time = time.time()

    print('Starting HT job: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if os.path.exists("./logs"):
        shutil.rmtree("./logs")
    os.makedirs("./logs")
    scheduler_log_file = open("./logs/scheduler.log", "w")

    workers = {i: xc.ServerProxy(ip) for i, ip in enumerate(worker_ips)}

    current_msts = [(mst_id, mst) for mst_id, mst in enumerate(initial_msts)]
    mst_evaluation_state = {}

    if os.path.exists(ckpt_root):
        shutil.rmtree(ckpt_root)

    for mst_id, mst in current_msts:
        ckpt_path = ckpt_root + "/" + str(mst_id) + "_" + uuid()
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        ckpt_path = ckpt_path + "/model"
        mst_evaluation_state[mst_id] = {"state": "RUNNING", "epoch": -1, "train_error": [], "train_loss": [],
                                        "valid_error": [],
                                        "valid_loss": [],
                                        "ckpt_path": ckpt_path,
                                        "mst": mst}
        log_file = open("./logs/" + str(mst_id) + ".log", 'a')
        log_message(log_file, "Checkpoint Path: " + ckpt_path + "\n")
        log_message(log_file, "MST: " + mst_identifier(mst) + "\n")

    if backend == 'tf':
        exec_fn_string = base64.b64encode(dill.dumps(tf_execute_helper, byref=False)).decode("ascii")
    elif backend == 'pytorch':
        exec_fn_string = base64.b64encode(dill.dumps(pytorch_execute_helper, byref=False)).decode("ascii")

    preload_fn_string = base64.b64encode(dill.dumps(preload_data_helper, byref=False)).decode("ascii")

    input_fn_string = base64.b64encode(dill.dumps(input_fn, byref=False)).decode("ascii")
    model_fn_string = base64.b64encode(dill.dumps(model_fn, byref=False)).decode("ascii")
    train_fn_string = base64.b64encode(dill.dumps(train_fn, byref=False)).decode("ascii")

    if preload_data_to_mem:
        # preload data into the worker memory
        preload_data(workers, input_fn_string, preload_fn_string, train_partitions, valid_partitions,
                     train_availability, valid_availability, scheduler_log_file, begin_time)

    # assume # train partitions = # valid. partitions
    P = len(train_partitions)
    W = len(workers)

    random.seed = 2019

    def _get_runnable_unit(epoch_units, w, availability, epoch_mst_execution_state, mode):
        random.shuffle(epoch_units)
        for idx, (mst_id, mst, partition) in enumerate(epoch_units):
            if availability[w][partition] == 1 and (epoch_mst_execution_state[mst_id] == False or mode == "VALID"):
                del epoch_units[idx]
                return mst_id, mst, partition
        return -1, -1, -1

    iteration = 0
    while len(current_msts) > 0:

        epoch_mst_evaluation_state = {mst_id: {"train_error": [], "train_loss": [], "valid_error": [], "valid_loss": []}
                                      for mst_id, mst in current_msts}

        for mode, availability, partitions in zip(["TRAIN", "VALID"], [train_availability, valid_availability],
                                                  [train_partitions, valid_partitions]):

            epoch_units = [(mst_id, mst, partition) for partition in range(P) for (mst_id, mst) in current_msts]
            epoch_mst_execution_state = {mst_id: False for mst_id, _ in current_msts}
            epoch_machine_state = [None for _ in range(W)]

            epoch_begin_time = time.time()
            while len(epoch_units) > 0 or sum([1 for x in epoch_machine_state if x is not None]) > 0:
                for w in [w for w in range(W) if w in workers]:

                    try:
                        if epoch_machine_state[w] is None:
                            mst_id, mst, p = _get_runnable_unit(epoch_units, w, availability, epoch_mst_execution_state, model)
                            if mst_id != -1:
                                exec_id = launch_job(workers[w],
                                                     mst_evaluation_state[mst_id]['epoch'] + 1,
                                                     [p],
                                                     mst_evaluation_state[mst_id]['ckpt_path'],
                                                     [partitions[p]],
                                                     input_fn_string,
                                                     model_fn_string, train_fn_string,
                                                     exec_fn_string, mst, mode == "TRAIN")
                                epoch_mst_execution_state[mst_id] = True
                                epoch_machine_state[w] = (mst_id, mst, p, exec_id)

                                message = "TIME: %d, EVENT: %s_LAUNCHED, ITERATION: %d, WORKER: %d, MST: %d, PARTITIONS: %s, EPOCH: %d, %s\n" % (
                                    time.time() - begin_time, mode, iteration, w, mst_id,
                                    "/".join([str(x) for x in [p]]),
                                    mst_evaluation_state[mst_id]['epoch'] + 1,
                                    mst_identifier(mst))
                                log_message(scheduler_log_file, message, print_message=True)
                        elif epoch_machine_state[w] is not None:
                            mst_id, mst, p, exec_id = epoch_machine_state[w]
                            completed, status = check_finished(workers[w], exec_id)
                            if completed:
                                epoch_mst_execution_state[mst_id] = False
                                epoch_machine_state[w] = None

                                log_file = open("./logs/" + str(mst_id) + ".log", 'a')
                                log_message(log_file, status["result"]["message"])

                                loss = status["result"]["loss"]
                                error = status["result"]["error"]

                                if mode == "TRAIN":
                                    epoch_mst_evaluation_state[mst_id]['train_loss'].extend(loss)
                                    epoch_mst_evaluation_state[mst_id]['train_error'].extend(error)
                                else:
                                    epoch_mst_evaluation_state[mst_id]['valid_loss'].extend(loss)
                                    epoch_mst_evaluation_state[mst_id]['valid_error'].extend(error)

                                message = "TIME: %d, EVENT: %s_COMPLETED, ITERATION: %d, WORKER: %d, MST: %d, PARTITIONS: %s, EPOCH: %d, %s\n" % (
                                    time.time() - begin_time, mode, iteration, w, mst_id,
                                    "/".join([str(x) for x in [p]]),
                                    mst_evaluation_state[mst_id]['epoch'] + 1,
                                    mst_identifier(mst))
                                log_message(scheduler_log_file, message, print_message=True)
                    except Exception as e:
                        print(e)
                        print('Worker {0} failure detected....'.format(str(w)))
                        # removing w from available workers
                        workers.pop(w, None)

                        # if there was any mst unit running, remove it back to the queue
                        if epoch_machine_state[w] is not None:
                            mst_id, mst, p, exec_id = epoch_machine_state[w]
                            print('MST {0} partition {1} moved back to queue....'.format(str(mst_id), str(p)))
                            epoch_units.append((mst_id, mst, p))
                            epoch_machine_state[w] = None
                            epoch_mst_execution_state[mst_id] = False

                        # starting from beginning
                        break

                # check failed workers are up again
                for w in range(W):
                    if w not in workers:
                        try:
                            #print('Checking worker {0}....'.format(str(w)))
                            con = xc.ServerProxy(worker_ips[w])
                            con.is_live()
                            workers[w] = con
                            epoch_machine_state[w] = None
                            print('Worker {0} back online....'.format(str(w)))
                            
                            if preload_data_to_mem:
                              # preload data into the worker memory
                              preload_data([workers[w]], input_fn_string, preload_fn_string,
                                           train_partitions, valid_partitions,
                                           [train_availability[w]], [valid_availability[w]], scheduler_log_file, begin_time)
                            
                        except Exception as e:
                            #print(e)
                            continue

                sys.stdout.flush()
                time.sleep(5)

            message = 'Iteration: {}, {} Elapsed Time: {}\n'.format(iteration, mode, time.time() - epoch_begin_time)
            log_message(scheduler_log_file, message, print_message=True)

        # update mst evaluation state
        mst_evaluation_state = update_mst_evaluation_state(epoch_mst_evaluation_state, mst_evaluation_state)

        # mst evaluation
        current_msts, mst_evaluation_state = evaluate_msts(mst_eval_fn, mst_evaluation_state, current_msts, ckpt_root)
        iteration += 1

    print('Total HT job time: ' + str(time.time() - begin_time))
