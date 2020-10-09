.. Cerebro documentation master file, created by
   sphinx-quickstart on Thu May  7 14:04:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Cerebro Documentation
=====================

``Cerebro`` is a data system for optimized deep learning model selection. It uses a novel parallel execution strategy
called **Model Hopper Parallelism (MOP)** to execute end-to-end deep learning model selection workloads in a more
resource-efficient manner.


Installation
------------

The best way to install the ``Cerebro`` is via pip.

.. highlight:: bash
.. code-block:: bash

    pip install -U cerebro-dl

Alternatively, you can git clone and run the provided Makefile script to install the master branch

.. highlight:: bash
.. code-block:: bash

    git clone https://github.com/ADALabUCSD/cerebro-system.git && cd cerebro-system && make

.. Note:: You MUST be running on **Python >= 3.6** with **Tensorflow >= 2.2** and **Apache Spark >= 2.4**



Table of Contents
-----------------

.. toctree::
   :maxdepth: 2


   about
   install
   quick_start
   api
   acknowledgement


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
