# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 22:38:27 2017

@author: Administrator
"""

import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn
import os
a = os.listdir("F:/project/spectrum-data")
df = pandas.DataFrame()
z=[]
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df1 = df[df.fc.between(825,835)]#1
    df1['fc'] = df1['fc']
    z.append(df1)


def projection(A,b):
    ####
    # return A*inv(AT*A)*AT*b
    ####
    AA = A.T.dot(A)
    w=np.linalg.inv(AA).dot(A.T).dot(b)
    print (w)
    return A.dot(w)

def projection1(A,b):
    ####
    # return A*inv(AT*A)*AT*b
    ####
    AA = A.T.dot(A)
    w=np.linalg.inv(AA).dot(A.T).dot(b)
    return w
theta=[]
for i in range(len(z)) :
    X = np.array(z[i]['fc'])
    Y = np.array(z[i]['E'])
    b = Y.reshape(Y.shape[0],1)
    m = []
    for i in range(5):
        m.append(X**(i))  
    A = np.array(m).T
    w = projection1(A,b)
    theta.append(w)
    yw = projection(A,b)
    yw.shape = (yw.shape[0],)
    plt.figure(figsize=(8,4))
    plt.plot(X,Y,color='m',linestyle='',marker='o',label=u"points")
    plt.plot(X,yw,color='r',linestyle='-',marker='.',label=u"fitted")
    plt.savefig('F:/project/operatordata/pic1/fitCdmaUp/%i.png'%i)
    plt.legend()
    plt.show()
    plt.figure()
    
    