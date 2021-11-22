#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from keras_tuner import HyperParameters

import autokeras as ak

from cerebro.nas.hphpmodel import HyperHyperModel


# In[2]:


feature_columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

input_node = [ak.StructuredDataInput() for c in feature_columns]
embeddings = [ak.StructuredDataBlock()(innode) for innode in input_node]
output_node = ak.Merge()([embeddings])
output_node = ak.ClassificationHead()(output_node)
am = HyperHyperModel(
    inputs=input_node, outputs=output_node, overwrite=True
)


# In[3]:


from pyspark.sql import SparkSession

# Build the SparkSession
spark = SparkSession.builder    .appName("Iris test")    .getOrCreate()

sc = spark.sparkContext

from cerebro.backend import SparkBackend
from cerebro.storage import LocalStore

backend = SparkBackend(spark_context=sc, num_workers=1)
store = LocalStore(prefix_path='/Users/zijian/Desktop/ucsd/cse234/project/cerebro-system/experiments')

am.resource_bind(
    backend=backend, 
    store=store,
    feature_columns=feature_columns,
    label_columns=['Species'],
    evaluation_metric='accuracy', 
)


# In[4]:


df = spark.read.csv("/Users/zijian/Desktop/ucsd/cse234/project/cerebro-system/Iris_clean.csv", header=True, inferSchema=True)

train_df, test_df = df.randomSplit([0.8, 0.2])
df.head(10)


# In[5]:


am.tuner_bind(
    tuner="randomsearch", 
    hyperparameters=None, 
    objective="val_loss",
    max_trials=20
)
am.fit(train_df, epochs=10)


# In[6]:


am.tuner.search_results


# In[7]:


am.tuner._finished


# In[8]:


am.tuner.oracle._tried_so_far


# In[ ]:




