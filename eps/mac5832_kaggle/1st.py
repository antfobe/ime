#!/usr/bin/python3

import pandas as pd
df = pd.read_csv('train.csv',encoding='latin1', dtype={'SourcePath': str}, );
print(list(df.columns.values))
df = df.drop(['y'], axis = 1);
print(list(df.columns.values))

import numpy as np
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
for col in df:
	df[col] = le.fit_transform(df[col]);

datanp = df.as_matrix();
X_train = datanp[:, 0:];
Y_train = datanp[:, 1];
df.head();

testdf = pd.read_csv('test.csv',encoding='latin1', dtype={'SourcePath': str}, );
##testdf.drop(['y'], axis = 1);
##testdf.DocumentType = testdf.DocumentType.astype(str);
testdf.head();

letest = preprocessing.LabelEncoder()
for col in testdf:
	testdf[col] = letest.fit_transform(testdf[col]);

testnp = testdf.as_matrix();
X_test = testnp[:, 0:];
##Y_test = testnp[:, 1];

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import svm
'''
text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42))]);
text_clf_svm = text_clf_svm.fit(X_train, Y_train);
'''

clf = svm.SVC();
clf.fit(X_train, Y_train);

##predicted_svm = text_clf_svm.predict(X_test);
predicted_svm = clf.predict(X_test);
print(predicted_svm);
'''
np.mean(predicted_svm == Y_test);

pd.concat([pd.Series(X_test),pd.Series(Y_test)],axis=1);

data.to_csv('outpoot2.csv');
'''
