# -*- coding: utf-8 -*-

# import modules
import pandas as pd
import numpy as np
from pyspark import SparkContext
from pyspark.ml import *
from pyspark.ml.classification import *
from pyspark.ml.feature import *
from pyspark.ml.param import *
from pyspark.ml.tuning import *
from pyspark.ml.evaluation import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import plotly.graph_objects as go 

# import modules for feature transformation
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF,StopWordsRemover,IDF,Tokenizer

# import modules for grid search
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


# setup SparkContext
sc = SparkContext('local[3]')
spark = SparkSession(sc)
sqlContext = SQLContext(sc)

# load data
data = pd.read_csv('train.csv')

# drop unnecessary columns
data = data.drop(['id','severe_toxic', 'obscene', 'threat', 'insult','identity_hate'], axis = 1)

# data analysis

# check data 
print(data.head(2))

d = data['toxic'].value_counts()
print(d[0])
print(data.info())


labels = ['not toxic','toxic']
values = data['toxic'].value_counts(normalize=True)
fig = go.Figure(data=[go.Pie(labels=labels,values=values)])
print(fig.show())


# create dataframe

# create fields and schema for creating dataframe from RDD
fields = [StructField('text', StringType(), True), StructField('label', ByteType(), True) ]
schema = StructType(fields)

# apply schema to the RDD
df = spark.createDataFrame(data, schema)
print(df.take(1))

# calculate ratio of toxic comments
toxic = df.filter(df.label == 1).count()
total_count = df.count()
toxic_ratio = toxic/total_count
print('The ratio of toxic comments : {:.3f}: {} out of {}'.format(toxic_ratio, toxic, total_count))


# split dataset into train & test data
# set 20% of the data for testing the model, 80% for training the model
train_set, test_set = df.randomSplit([0.8, 0.2], 123)
print("Total document count:",df.count())
print("Training-set count:",train_set.count())
print("Test-set count:",test_set.count())

# create pipeline for feature transformation &  model tuning

# tokenize into words
tokenizer = Tokenizer().setInputCol("text").setOutputCol("words")

# remove stopwords
remover= StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(False)

# for each sentence (bag of words), use HashingTF to hash the sentence into a feature vector
hashingTF = HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")

# create TF_IDF features
idf = IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)

# create a Logistic regression model
lr = LogisticRegression()

# streamline all above steps into a pipeline
pipeline = Pipeline(stages=[tokenizer,remover,hashingTF,idf, lr])


# train model and predict results

# perform grid search looking for the best parameters and the best models
paramGrid = ParamGridBuilder()\
    .addGrid(hashingTF.numFeatures,[1000,5000,10000])\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.3, 0.6])\
    .build()
tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=BinaryClassificationEvaluator().setMetricName('areaUnderPR'),
                           trainRatio=0.8)
                           # set area under precision-recall curve as the evaluation metric - 80% of data will be used for training, 20% for validation



# run TrainValidationSplit and choose the best set of parameters
model = tvs.fit(train_set)

# make predictions
train_prediction = model.transform(train_set)
test_prediction = model.transform(test_set)


# report accuracy

# caculate the accuracy score for the best model 
correct = test_prediction.filter(test_prediction.label == test_prediction.prediction).count()  
accuracy = correct/test_prediction.count()
print('Accuracy {:.2%} data items: {}, correct: {}'.format(accuracy, test_prediction.count(), correct))

