Install
-------

### Basic Installation

The best way to install the ``Cerebro`` is via pip.

    pip install -U cerebro-dl

Alternatively, you can git clone and run the provided Makefile script to install the master branch.

    git clone https://github.com/ADALabUCSD/cerebro-system.git && cd cerebro-system && make

You MUST be running on **Python >= 3.6** with **Tensorflow >= 2.2** and **Apache Spark >= 2.4**


### Spark Cluster Setup

As deep learning workloads tend to have very different resource requirements
from typical data processing workloads, there are certain considerations
for DL Spark cluster setup.

#### GPU training

For GPU training, one approach is to set up a separate GPU Spark cluster
and configure each executor with ``# of CPU cores`` = ``# of GPUs``. This can
be accomplished in standalone mode as follows:

```bash
$ echo "export SPARK_WORKER_CORES=<# of GPUs>" >> /path/to/spark/conf/spark-env.sh
$ /path/to/spark/sbin/start-all.sh
```

This approach turns the ``spark.task.cpus`` setting to control # of GPUs
requested per process (defaults to 1).

The ongoing [SPARK-24615](https://issues.apache.org/jira/browse/SPARK-24615) effort aims to
introduce GPU-aware resource scheduling in future versions of Spark.

#### CPU training
For CPU training, one approach is to specify the ``spark.task.cpus`` setting
during the training session creation:

```python
conf = SparkConf().setAppName('training') \
    .setMaster('spark://training-cluster:7077') \
    .set('spark.task.cpus', '16')
spark = SparkSession.builder.config(conf=conf).getOrCreate()
```

This approach allows you to reuse the same Spark cluster for data preparation
and training.
