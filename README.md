Cerebro
=======

 
``Cerebro`` is a resource-efficient model selection system for deep learning.


Installation
------------

The best way to install the ``Cerebro`` is via pip.

    pip install -U cerebro

Alternatively, you can git clone and run the provided Makefile script

    git clone https://github.com/ADALabUCSD/cerebro-system.git && cd cerebro-system && make

You MUST be running on **Python >= 3.5** with **Tensorflow >= 2.0** and **Apache Spark >= 2.4**


Documentation
-------------

Detailed documentation about the system can be found [here](https://adalabucsd.github.io/cerebro-system/).


Acknowledgement
---------------
We used the following projects when building Cerebro.
- [Horovod](https://github.com/horovod/horovod): Cerebro's Apache Spark implementation uses code from the Horovod's
 implementation for Apache Spark.
- [Petastorm](https://github.com/uber/petastorm): We use Petastorm to read Apache Parquet data from remote storage
 (e.g., HDFS)  
 
Cite
----
```latex
@inproceedings{nakandala2019cerebro,
  title={Cerebro: Efficient and Reproducible Model Selection on Deep Learning Systems},
  author={Nakandala, Supun and Zhang, Yuhao and Kumar, Arun},
  booktitle={Proceedings of the 3rd International Workshop on Data Management for End-to-End Machine Learning},
  pages={1--4},
  year={2019}
}

```