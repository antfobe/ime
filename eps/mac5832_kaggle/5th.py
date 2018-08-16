#!/usr/bin/python3

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing

# function to parse dataframe lines to tf standard (assumes all values are numbers)
# returns 
def _parse_dataframe(df):
    # Decode the line into its fields
    for index, line in df.iterrows():
        fields = tf.decode_csv(line, np.zeros(len(df.columns)))

        # Pack the result into a dictionary
        features = dict(zip(list(df),fields))

        # Separate the label from the features
        label = features.pop('label')

    return features, label

def train_input_fn(features, labels):
    """An input function for training"""
    batch_size = 9061;
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

## bring data in
df = pd.read_csv('train.csv',encoding='latin1', dtype={'SourcePath': str}, );

le = preprocessing.LabelEncoder()
for col in df:
	df[col] = le.fit_transform(df[col]);

train_y = {};
#train_y = _parse_dataframe(df['y']);
train_x = {};
#train_x = _parse_dataframe(df.drop(['y'], axis = 1));

#scaler = preprocessing.StandardScaler()
#scaler.fit(X_train);
#X_train = scaler.transform(X_train);

testdf = pd.read_csv('test.csv',encoding='latin1', dtype={'SourcePath': str}, );

#letest = preprocessing.LabelEncoder()
for col in testdf:
	testdf[col] = le.fit_transform(testdf[col]);

testnp = testdf.values;
test = testnp[:, 0:];
#test = scaler.transform(test);

labels = list(df);
# Build a DNN with 4 hidden layers and 12,8,5,3 nodes in the hidden layers.
classifier = tf.estimator.DNNClassifier(
    feature_columns=train_x,
    hidden_units=[12, 8, 5, 3],
    # The model must choose between 2 classes.
    n_classes=2)

# Train the Model.
classifier.train(
    input_fn=lambda:train_input_fn(_parse_dataframe(df)))

# Evaluate the model.
eval_result = classifier.evaluate(
    input_fn=lambda:train_input_fn(_parse_dataframe(df)))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result));

##np.savetxt('tf-dn-submission.csv', X=np.vstack((testdf.values[:,0], pred)).T, fmt='%d', delimiter=',', header='id,y');
