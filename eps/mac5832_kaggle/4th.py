#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier


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

## function to plot fits
def plot_on_dataset(X, y, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ## ax.set_title(name)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    mlps = []
    if name == "digits":
        # digits is larger but converges fairly quickly
        max_iter = 15
    else:
        max_iter = 400

    best = 0;
    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0,
                            max_iter=max_iter, **param)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
        if mlp.score(X,y) > best:
            best = mlp.score(X,y);
            pred = mlp.predict(test);

    ##for mlp, label, args in zip(mlps, labels, plot_args):
    ##        ax.plot(mlp.loss_curve_, label=label, **args)
    np.savetxt('nn-mlp-submission.csv', X=np.vstack((testdf.values[:,0], pred)).T, fmt='%d', delimiter=',', header='id,y');

## set multiple params for attempts at best fit
params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01},
          {'solver': 'lbfgs', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.0001}]

labels = ["constant learning-rate", "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam", "lbfgs"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'yellow', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'}]


clf = MLPClassifier(solver='lbfgs', alpha=1.0, hidden_layer_sizes=(12, 8, 5, 3), random_state=1)
clf.fit(X_train, Y_train)                         
MLPClassifier(activation='relu', alpha=1.0, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(12, 8, 5, 3), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

pred = clf.predict(test);
print("Training set score: %f" % clf.score(X_train, Y_train));
print("Test set score: %f" % clf.score(test, pred));
np.savetxt('nn-submission.csv', X=np.vstack((testdf.values[:,0], pred)).T, fmt='%d', delimiter=',', header='id,y');
'''
fig, axes = plt.subplots(2, 2, figsize=(15, 10));
plot_on_dataset(X=X_train, y=Y_train, ax=axes.ravel(), name='kaggle');
'''
##fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center");
##plt.show();
