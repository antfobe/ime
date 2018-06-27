import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation,Dense,Flatten,Convolution1D,Lambda
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import backend,optimizers
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

seed = 13
np.random.seed(seed)


# load/treat data
df = pd.read_csv('train.csv',encoding='latin1', dtype={'SourcePath': str}, );
testdf = pd.read_csv('test.csv',encoding='latin1', dtype={'SourcePath': str}, );

le = preprocessing.LabelEncoder()
for col in df:
	if col in ["age", "job", "marital", "default", "housing", "loan", "contact", "poutcome", "education"]: 
		df[col] = le.fit_transform(df[col]);
		if col in testdf:
			testdf[col] = le.fit_transform(testdf[col]);

df = df.drop(['id'], axis = 1);
testdf = testdf.drop(['id'], axis = 1);

datanp = df.values;
Y = datanp[:, 0].astype(int);
datanp = df.drop(['y'], axis = 1).values;
X = datanp[:, 0:].astype(float);
dummy_y = np_utils.to_categorical(Y, num_classes=2);
T = testdf.values[:,0:].astype(float);

def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(95, input_dim=19, kernel_initializer='normal', activation='relu'))
	model.add(Dense(190, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	adad = optimizers.Adadelta(lr=1.1, rho=0.95, epsilon=None, decay=0.0)
	sgd = optimizers.SGD(lr=.0011, momentum=0.0, decay=0.0, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer=adad, metrics=['accuracy'])
	#model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	#model.fit(X, Y, epochs=100, batch_size=df.shape[0])
	return model

# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=128, batch_size=df.shape[0], verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''
# predict on test set
pipeline.fit(X, Y)
pred = pipeline.predict(T)
print(pred.shape, "-", np.arange(1, pred.size + 1, dtype=int).reshape(pred.size,1).shape)
#np.savetxt('nn-keras-submission.csv', X=np.dstack((np.arange(1, pred.size + 1), pred)), fmt='%d,%d', delimiter=',', header='id,y', comments='');
with open('nn-keras-submission.csv', 'w') as f:
	f.write('\"id\",\"y\"\n')
	itr = 0
	for value in np.nditer(pred):
		f.write(str(itr) + ',' + str(value) + '\n')
		itr = itr + 1
'''
