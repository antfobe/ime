#!/usr/bin/python3

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing

## bring data in
df = pd.read_csv('train.csv',encoding='latin1', dtype={'SourcePath': str}, );

le = preprocessing.LabelEncoder()
for col in df:
	df[col] = le.fit_transform(df[col]);

datanp = df.values;
Y_train = datanp[:, 1];
datanp = df.drop(['y'], axis = 1).values;
X_train = datanp[:, 0:];

scaler = preprocessing.StandardScaler()
scaler.fit(X_train);
X_train = scaler.transform(X_train);

testdf = pd.read_csv('test.csv',encoding='latin1', dtype={'SourcePath': str}, );

letest = preprocessing.LabelEncoder()
for col in testdf:
	testdf[col] = letest.fit_transform(testdf[col]);

testnp = testdf.values;
test = testnp[:, 0:];
test = scaler.transform(test);

labels = list(df);
# Build a DNN with 4 hidden layers and 12,8,5,3 nodes in the hidden layers.
classifier = tf.estimator.DNNClassifier(
    feature_columns=X_train,
    hidden_units=[12, 8, 5, 3],
    # The model must choose between 2 classes.
    n_classes=2)

# Train the Model.
classifier.train(
    input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
    steps=args.train_steps)

# Evaluate the model.
eval_result = classifier.evaluate(
    input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result));

##np.savetxt('tf-dn-submission.csv', X=np.vstack((testdf.values[:,0], pred)).T, fmt='%d', delimiter=',', header='id,y');
