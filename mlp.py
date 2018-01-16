#!/usr/bin/env python
# coding: utf-8
# based on https://github.com/andersbll/nnet
# modified by heibanke
"""
mlp: use nn to identify digital number
"""
import numpy as np
from matplotlib import pyplot as plt

from nnet.layers import Linear, Activation, MSECostLayer
from nnet.neuralnetwork import neuralnetwork

#import sklearn.datasets

#digits = sklearn.datasets.load_digits()









X_train = r
X_train /= np.max(X_train)
y_train = label



n_classes = np.unique(y_train).size

# Setup multi-layer perceptron 
nn = neuralnetwork(
    layers=[
        Linear(
            n_out=4,
            weight_scale=0.1,
        ),
        Activation('relu'), 
        Linear(
            n_out=4,
            weight_scale=0.1,
        ),
        Activation('relu'), 
        Linear(
            n_out=4,
            weight_scale=0.1,
        ),
        Activation('relu'), 
        Linear(
            n_out=n_classes,
            weight_scale=0.1,
        ),
        Activation('tanh'),        
    ],
    cost=MSECostLayer(),
)

TRAIN_NUM = len(y_train)*2/3

# Train neural network
print 'Training neural network'
nn.train(X_train, y_train, learning_rate=0.00001, max_iter=50000, batch_size=4)
#nn.train_scipy(X_train, y_train)
# Evaluate on training data
error = nn.error(k, l)
print ('Training error rate: %.4f' % error)


plot_decision_regions(X_train,y_train,clf=nn)