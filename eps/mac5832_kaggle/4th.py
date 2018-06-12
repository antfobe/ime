#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('train.csv',encoding='latin1', dtype={'SourcePath': str}, );
print(list(df.columns.values))

le = preprocessing.LabelEncoder()
for col in df:
	df[col] = le.fit_transform(df[col]);

datanp = df.values;
Y_train = datanp[:, 1];
df = df.drop(['y'], axis = 1);
datanp = df.values;
X_train = datanp[:, 0:];
print(X_train[0,:]);

testdf = pd.read_csv('test.csv',encoding='latin1', dtype={'SourcePath': str}, );

letest = preprocessing.LabelEncoder()
for col in testdf:
	testdf[col] = letest.fit_transform(testdf[col]);

testnp = testdf.values;
test = testnp[:, 0:];
print(test[0,:]);

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 8, 5, 3), random_state=1)
clf.fit(X_train, Y_train)                         
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

print(clf.predict(test))
