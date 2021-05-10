
# Background
We provide an example of how we used `cerebro-standalone` to scale the model selection of behavior classification models that were released in the [DeepPostures](https://github.com/ADALabUCSD/DeepPostures/tree/master/MSSE-2021) library. `cerebro-standalone` is a bare-bones implementation of model hopper parallelism that runs on a shared file system such as NFS. It requires users to provide four functions: input data loading function, model definition function, model training function, and a model evaluation function. Also, it does not provide features for data shuffling and data partitioning.


This example is provided for reproducibility purposes. For new users, we recommend using the `cerebro-spark` version instead of the `cerebro-standalone` version.

# Steps
## 1. Preparing the Training Data
1. Use the [DeepPostures](https://github.com/ADALabUCSD/DeepPostures/tree/master/MSSE-2021) library to generate the pre-processed data. Store these pre-processed files in a shared directory (e.g., NFS mount), which is accessible from all workers. Pre-processed data is organized by directories each corresponding to a different human subject. Each directory contains multiple files corresponding to each device-wear date.
2. Create partition definition files, which provide paths to pre-processed data directories. If you have `n` workers, create `n` train and validation partitions. Example partition definition files are provided in ./data/train and ./data/valid directories.
3. Store the partition definition files also in a shared directory accessible by all workers.
4. Update the data partition definition section in the `model_selection.py:main()` method to point to your definition files.

## 2. (Optional) Modify the Workload
1. The workload is defined using TensorFlow V1 APIs and runs in lazy mode. We provide 4 ready-to-use functions that capture the workload: `input_fn`, `model_fn`, `train_fn`, and `eval_fn`.
2. `input_fn` takes in a path value of a data partition definition file and loads all the data in that partition.
3. `model_fn` takes in a parameter setting and creates a CNN-BiLSTM model and a initialized model training. The CNN operates on 10s of tri-axial acceleration values window, which are at 10Hz (altogether 100 tri-axial acceleration values). If you want to change these values, change them in the `model_fn`.
4. `train_fn` handles the training for one epoch and returns the training loss and error. For validation it will be called with `train=False` and only the error and loss will be calculated without any training.
5. `eval_fn` implements the model selection algorithm. We implement grid search, which trains all models for 10 epochs. If a model has been trained for 10 epochs it is added to the stop list. If you want to train models for more epochs, increase 10 to a larger value.

More details on these four functions can be found in the Appendix A of [Cerebro Technical Report](https://adalabucsd.github.io/papers/TR_2020_Cerebro.pdf)

## 3. Running the workload
1. You need to first install cerebro on all your workers. This can be done by running the command `pip install cerebro-dl`.
2. Start cerebro standalone workers on all your workers. This can be done using the command: `cerebro-standalone-worker`. You can also specify a hostname and a port on which this worker should listen on by passing `--hostname <hostname>` and `--port <port>` options. By default it will listen on `0.0.0.0:7777`.
3. Update the worker hostname and ports in the workers definition of the `model_selection.py:main()` method.
4. Run the script by executing the command: `python model_selection.py`.
