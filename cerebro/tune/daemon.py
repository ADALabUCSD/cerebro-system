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
import os
import time
import logging
import traceback
import tensorflow as tf
from .base import log_hp_to_tensorboard, log_epoch_metrics_to_tensorboard, estimator_gen_fn_wrapper, update_model_results
from ..commons.constants import *
from ..db.dao import *
from ..storage import LocalStore, HDFSStore
from importlib import import_module
from sqlalchemy import and_


def sub_epoch_scheduler(app, db, backend, inter_epoch_wait_time=5, verbose=True):
    """
    Sub-epoch scheduling daemon. Reads trainable model configs from the database and runs them on the provided backend.
    :param app: Flask applicarion.
    :param db: SQLAlchemy DB object.
    :param backend: Cerebro backend object
    :param inter_epoch_wait_time:
    :param verbose:
    """

    with app.app_context():
        while not exit_event.is_set():
            all_models = all_models = Model.query.filter(and_(Model.status.in_([CREATED_STATUS, RUNNING_STATUS]), Model.max_train_epochs > Model.num_trained_epochs)).all()
            if all_models is not None and len(all_models) > 0:
                estimators = []
                estimator_results = {}
                all_stores = {}
                all_labels = {}
                all_features = {}
                for m in all_models:
                    try:
                        exp_obj = Experiment.query.filter(Experiment.id == m.exp_id).one()
                        data_store_prefix_path = exp_obj.data_store_prefix_path

                        if data_store_prefix_path.startswith('hdfs://'):
                            store = HDFSStore(prefix_path=data_store_prefix_path)
                        else:
                            store = LocalStore(prefix_path=data_store_prefix_path)
                    
                        param = {}
                        for d in m.param_vals:
                            if d.dtype == DTYPE_FLOAT:
                                param[d.name] = float(d.value)
                            elif d.dtype == DTYPE_INT:
                                param[d.name] = int(d.value)
                            else:
                                param[d.name] = d.value

                        mod, f = exp_obj.executable_entrypoint.split(':')
                        mod = import_module(mod)
                        estimator_gen_fn = getattr(mod, f)
                        
                        features, labels = exp_obj.feature_columns.split(','), exp_obj.label_columns.split(',')
                        est = estimator_gen_fn_wrapper(estimator_gen_fn, param, features, labels, store, verbose)
                        est.setRunId(m.id)
                        est.setEpochs(m.num_trained_epochs)

                        # Creating model checkpoint
                        remote_store = store.to_remote(est.getRunId())
                        with remote_store.get_local_output_dir() as run_output_dir:
                            tf.compat.v1.reset_default_graph
                            model = est._compile_model(est._get_keras_utils())

                            if m.warm_start_model_id is not None and not est._has_checkpoint(m.id):
                                # https://www.tensorflow.org/guide/keras/save_and_serialize#apis_for_in-memory_weight_transfer
                                remote_store2 = store.to_remote(m.warm_start_model_id)
                                with remote_store2.get_local_output_dir() as run_output_dir2:
                                    model2 = est._compile_model(est._get_keras_utils())
                                    model.set_weights(model2.get_weights())

                                warm_start_model = Model.query.filter(Model.id == m.warm_start_model_id).one()
                                db.session.refresh(m)
                                m.num_trained_epochs = warm_start_model.num_trained_epochs
                                db.session.commit()

                                est.setEpochs(m.num_trained_epochs)

                                for metric in warm_start_model.metrics:
                                    new_metric = Metric(m.id, metric.name, [float(x) for x in metric.values.split(",")])
                                    db.session.add(new_metric)
                                    db.session.commit()

                            ckpt_file = os.path.join(run_output_dir, remote_store.checkpoint_filename)
                            model.save(ckpt_file)
                            remote_store.sync(run_output_dir)
                            tf.compat.v1.reset_default_graph

                        estimators.append(est)
                        all_stores[est.getRunId()] = store
                        all_features[est.getRunId()] = features
                        all_labels[est.getRunId()] = labels
                        
                        if m.status == CREATED_STATUS:
                            db.session.refresh(m)
                            m.status = RUNNING_STATUS
                            db.session.commit()
                            # Log hyperparameters to TensorBoard
                            log_hp_to_tensorboard([est], [param], store, verbose)

                        estimator_results[m.id] = {}
                        for metric in m.metrics:
                            estimator_results[m.id][metric.name] = [float(x) for x in metric.values.split(',')]
                    except Exception as e:
                        logging.error(traceback.format_exc())
                        db.session.refresh(m)
                        m.status = FAILED_STATUS
                        m.exception_message = str(traceback.format_exc())
                        db.session.commit()

                # Trains all the models for one epoch. Also performs validation
                epoch_results = backend.train_for_one_epoch(estimators, all_stores, all_features, all_labels)
                update_model_results(estimator_results, epoch_results)

                epoch_results = backend.train_for_one_epoch(estimators, all_stores, all_features, all_labels, is_train=False)
                update_model_results(estimator_results, epoch_results)

                log_epoch_metrics_to_tensorboard(estimators, estimator_results, all_stores, verbose)

                for m in all_models:
                    est_results = estimator_results[m.id]
                    # Refresh to sync any model stop requests from the db
                    db.session.refresh(m)
                    metrics = m.metrics.all()
                    if len(metrics) == 0:
                        for k in est_results:
                            db.session.add(Metric(m.id, k, est_results[k]))
                    else:
                        for k in est_results:
                            metric = [metric for metric in metrics if metric.name == k][0]
                            metric.values = ",".join(["{:.4f}".format(x) for x in est_results[k]])
                    db.session.commit()

                for m in all_models:
                    # Refresh to sync any model stop requests from the db
                    db.session.refresh(m)
                    m.num_trained_epochs += 1
                    if m.num_trained_epochs >= m.max_train_epochs:
                        m.status = COMPLETED_STATUS
                    db.session.commit()

            # inter-epoch waiting
            exit_event.wait(inter_epoch_wait_time)
