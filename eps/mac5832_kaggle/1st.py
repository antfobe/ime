#!/usr/bin/python3

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn import svm
df = pd.read_csv('train.csv',encoding='latin1', dtype={'SourcePath': str}, );
df = df.drop(['y'], axis = 1);
print(list(df.columns.values))


le = preprocessing.LabelEncoder()
for col in df:
	df[col] = le.fit_transform(df[col]);

datanp = df.as_matrix();
X_train = datanp[:, 0:];
Y_train = datanp[:, 1];
print(X_train[0,:], X_train[1,:]);

testdf = pd.read_csv('test.csv',encoding='latin1', dtype={'SourcePath': str}, );
testdf.head();

letest = preprocessing.LabelEncoder()
for col in testdf:
	testdf[col] = letest.fit_transform(testdf[col]);

testnp = testdf.as_matrix();
X_test = testnp[:, 0:];
print(X_test[0,:], X_test[1,:]);

for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, Y_train)

    plt.figure(fig_num)
    plt.clf()
    plt.scatter(datanp[:, 0], datanp[:, 1], c=Y_train, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

    # Circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = datanp[:, 0].min()
    x_max = datanp[:, 0].max()
    y_min = datanp[:, 1].min()
    y_max = datanp[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.title(kernel)
plt.show()
'''
clf = svm.SVC();
clf.fit(X_train, Y_train);

predicted_svm = clf.predict(X_test);
with open('submission.csv', 'w') as submission:
	print('id,y', file = submission);
	for idx in np.nditer(predicted_svm):
		print(str(idx) + ','+ str(int(predicted_svm[idx])), file = submission);
'''
