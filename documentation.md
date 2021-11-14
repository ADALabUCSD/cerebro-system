## Documentation of Autokeras Extension on Cerebro

Welcome to our document! 

This project is an [AutoKeras](https://autokeras.com/) extension on [Cerebro](https://adalabucsd.github.io/cerebro-system/), 
which provides easy-to-use APIs for Neural Architecture Searching of deep learning models in a distributed manner. 
If you are new to AutoKeras or Cerebro), please refer to the _Getting Started_ section below.

### Getting Started

This project is an extension of Cerebro, in order to utilize the high throughput features of Cerebro to accerlerate neural
architecture searching process. Our APIs are identical to AutoKeras, yet generates a set of trials to run simutaneously.
To start with, you need to create a SparkSession for the Cerebro backend.  

```python
from pyspark.sql import SparkSession

# Build the SparkSession
spark = SparkSession.builder \
   .master("local") \
   .appName("Example") \
   .config("spark.executor.memory", "1gb") \
   .getOrCreate()
   
sc = spark.sparkContext
```

Then pass the SparkSession to the Cerebro to create a SparkEnd object for model training later.

```python
from cerebro.backend import SparkBackend
from cerebro.storage import LocalStore

backend = SparkBackend(spark_context=sc, num_workers=1)
store = LocalStore(prefix_path='Your data directory')
```

Then create a HyperModel using our API, and bind the resources to the hypermodel. You could assign the input and output
structure of the hypermodel.

```python
from cerebro.nas.hphpmodel import HyperHyperModel
import autokeras as ak

input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node1 = ak.ConvBlock()(output_node)
output_node2 = ak.ResNetBlock(version="v2")(output_node)
output_node = ak.Merge()([output_node1, output_node2])
output_node = ak.ClassificationHead()(output_node)

am = HyperHyperModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=1000
)

am.resource_bind(backend=backend, store=store)
```

Or to make things easier, you could only assign the output head type to the hypermodel and wait for the NAS algorithm to 
find the optimal neural architecture for you.

```python
from cerebro.nas.hphpmodel import HyperHyperModel
import autokeras as ak

input_node = ak.ImageInput()
output_node = ak.ClassificationHead()

am = HyperHyperModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=1000
)

am.resource_bind(backend=backend, store=store)
```

Then create a HyperParameters object and you could fix hyperparameters on your own will. After you create the 
hyperparameters set, bind it to the hypermodel.

```python
from tensorflow.keras.datasets import mnist
from keras_tuner import HyperParameters

hp = HyperParameters()
hp.Fixed("learning_rate", value=0.0001)
am.tuner_bind("hyperband", hyperparameters=hp)
```

Then we can start training with Spark DataFrame. So far you have to build a spark pipeline for data preprocessing.

```python
df = spark.read.format("libsvm").load("sample_libsvm_data.txt").repartition(8)
train_df, test_df = df.randomSplit([0.8, 0.2])

am.fit(train_df)

predict_result = am.predict(test_df)
best_model = am.export_model()
```

You can read the summary by calling the `search_space_summary()` method.

```python
am.tuner.search_space_summary()
```


### API references

Our APIs are in the `cerebro.nas` module.

#### 1. HyperHyperModel class

The complete path for `HyperHyperModel` class is `cerebro.nas.hphpmodel.HyperModel`.

##### Parameters for constructor
1. inputs (autokeras.nodes.Input): The input type of the HyperModel
2. outputs (autokeras.nodes.Output): The output type of the HyperModel
3. overwrite (bool): 
4. seed (Optional[int]): The seed for random calculation, default None

##### Methods

1. resource_bind()

2. tuner_bind()

3. fit()

4. predict()

5. evaluate()

6. export_model()


For other APIs you may need, please check out:

1. [Cerebro documentation](https://adalabucsd.github.io/cerebro-system/quick_start.html)
2. [AutoKeras documentation](https://autokeras.com/image_classifier/)
3. [KerasTuner documentation](https://keras.io/api/keras_tuner/)


### Dependencies

This is the dependencies of our project so far, 

```
h5py>=2.9
numpy
petastorm==0.9.0
pyarrow==0.16.0
cloudpickle
hyperopt>=0.2.3
transformers>=3.3.1
flask-restplus>=0.9.2
Flask-SQLAlchemy>=2.1
tensorflow>=2.2
keras_tuner
autokeras
```

To set up the environment quickly, please make use of the `requirements.txt` file in the directory,

```bash
pip3 install -r requirements.txt
```
