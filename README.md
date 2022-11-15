Cerebro
=======
 
``Cerebro`` is a data system for optimized deep learning model selection. It uses a novel parallel execution strategy
called **Model Hopper Parallelism (MOP)** to execute end-to-end deep learning model selection workloads in a more 
resource-efficient manner. Detailed technical information about ``Cerebro`` can be found in our 
[Technical Report](https://adalabucsd.github.io/papers/TR_2020_Cerebro.pdf).


Install
-------

**Prerequisites:** You MUST be running on **Python >= 3.6** with **Tensorflow >= 2.3** (note version 2.3 and >=2.9.1 are both known to work, but version 2.4 - 2.5 are not working) and **Apache Spark >= 2.4**. You will need to install these separately, and you will also need to install pyspark with a matching version of your Spark. For most users, these (except for Spark, which you will need to follow their instructions) can be installed by

```bash
pip install tensorflow==2.3
```
and

```bash
pip install pyspark==<your spark version>
```

It's worth mentioning pyspark itself can be run in local/single-node mode without Spark installed. If you are just checking out/not using a cluster, then you can run 
```bash
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk
pip install pyspark==3.2.0
```
This alone should be sufficient for running the examples, but remember, to utilize a cluster with multiple machines, you will need Spark eventually. 

**Cerebro:** The best way to install the ``Cerebro`` is via pip (may not contain the latest changes).

```bash
pip install -U cerebro-dl
```

Alternatively, you can git clone and run the provided Makefile script

```bash
git clone https://github.com/ADALabUCSD/cerebro-system.git && cd cerebro-system && make
```

Quick Start
-------------
There are three examples in increasing complexity.

1. Run the unit tests:
```bash
python -m pytest
```

2. Run a bare minimum model selection example:
```bash
cd examples
python dummy_model_selection.py
```

3. Run an end-to-end example:
```bash
cd examples
wget http://files.fast.ai/part2/lesson14/rossmann.tgz
tar zxvf rossmann.tgz
python rossmann_model_selection.py
```

Documentation
-------------

Detailed documentation about the system can be found [here](https://adalabucsd.github.io/cerebro-system/).


Acknowledgement
---------------
This project was/is supported in part by a Hellman Fellowship, the NIDDK of the NIH under award number R01DK114945, and an NSF CAREER Award.

We used the following projects when building Cerebro.
- [Horovod](https://github.com/horovod/horovod): Cerebro's Apache Spark implementation uses code from the Horovod's
 implementation for Apache Spark.
- [Petastorm](https://github.com/uber/petastorm): We use Petastorm to read Apache Parquet data from remote storage
 (e.g., HDFS)  
 
Publications
------------
If you use this software for research, plase cite the following papers:

```latex
@inproceedings{nakandala2019cerebro,
  title={Cerebro: Efficient and Reproducible Model Selection on Deep Learning Systems},
  author={Nakandala, Supun and Zhang, Yuhao and Kumar, Arun},
  booktitle={Proceedings of the 3rd International Workshop on Data Management for End-to-End Machine Learning},
  pages={1--4},
  year={2019}
}

```
