#!/usr/bin/python
# -*- coding: utf-8 -*-
#copyRight by heibanke 

"""
plot_decision_regions come from:
https://github.com/rasbt/mlxtend
"""

import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plot_func import plot_decision_regions
from nnet.layers import Linear, Activation, MSECostLayer
from nnet.neuralnetwork import neuralnetwork


if sys.argv[1]=='circle':
    # circle data
    t=np.linspace(0,6*np.pi,100)

    x1 = np.array([3+((np.random.rand(len(t))-0.5)+1)*np.sin(t),3+((np.random.rand(len(t))-0.5)+1)*np.cos(t)])
    x2 = np.array([3+((np.random.rand(len(t))-0.5)+3)*np.sin(t),3+((np.random.rand(len(t))-0.5)+3)*np.cos(t)])
    y1 = np.zeros(100)
    y2 = np.ones(100)

elif sys.argv[1]=='cos':

    # cos data
    t=np.linspace(0,2*np.pi,100)

    x1 = np.array([t,3+np.cos(t)])
    x2 = np.array([t,2+np.cos(t)])
    y1 = np.zeros(100)
    y2 = np.ones(100)
else:
    assert False,"add para circle or cos"  

X=np.concatenate((x1.T,x2.T))
y=np.concatenate((y1,y2))


#init and train
nn = neuralnetwork(
    layers=[
        Linear(
            n_out=2,
            weight_scale=0.1,
        ),
        Activation('sigmoid'), 
        Linear(
            n_out=2,
            weight_scale=0.1,
        ),
        Activation('sigmoid'),        
    ],
    cost=MSECostLayer(),
)

# Train neural network
print 'Training neural network'
nn.train(X, y, learning_rate=0.01, max_iter=2000, batch_size=200)

plot_decision_regions(X,y,clf=nn)

plt.show()