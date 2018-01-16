# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:59:16 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# load dataset

X = r
Y = label

# encode class values as integers
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
# convert integers to dummy variables (one hot encoding)
dummy_y = np_utils.to_categorical(encoded_Y)

# define model structure
def baseline_model():
    model = Sequential()
    model.add(Dense(output_dim=1200, input_dim=3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=6, input_dim=1200, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=1000, batch_size=256)
# splitting data into training set and test set. If random_state is set to an integer, the split datasets are fixed.
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3, random_state=0)
estimator.fit(X_train, Y_train,nb_epoch=5000)
# make predictions
pred = estimator.predict(X_test)

# inverse numeric variables to initial categorical labels
init_lables = encoder.inverse_transform(pred)

# k-fold cross-validate
seed = 42
np.random.seed(seed)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)