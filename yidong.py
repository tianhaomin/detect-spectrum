# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:05:55 2017

@author: Administrator
"""

import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn
import os







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



  
    theta1=[]
    
        X = np.array(df1['fc'])
        Y = np.array(df1['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(2):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        #w = w.T.tolist()
        theta1.append(w[1])
        yw = projection(A,b)
        yw.shape = (yw.shape[0],)
        
        
        
        
        
        theta2 = []
        X1 = np.array(df2['fc'])
        Y1 = np.array(df2['E'])
        b = Y1.reshape(Y1.shape[0],1)
        m = []
        for i in range(2):
            m.append(X**(i))  
        A = np.array(m).T
        w1 = projection1(A,b)
        #w = w.T.tolist()
        theta2.append(w1[1])
        yw1 = projection(A,b)
        yw1.shape = (yw1.shape[0],)





        plt.figure(figsize=(8,4))
        plt.plot(X,Y,color='m',linestyle='',marker='o',label=u"points")
        plt.plot(X,yw,color='r',linestyle='-',marker='.',label=u"233.6-0.292X")
        plt.plot(X1,Y1,color='m',linestyle='',marker='o',label=u"points")
        plt.plot(X1,yw1,color='g',linestyle='-',marker='.',label=u"389.9-0.477X")
        #plt.savefig('F:/project/operatordata/pic1/fitCdmaUp/%i.png'%i)
        plt.legend()
        plt.show()
        plt.figure()







a = os.listdir("F:/project/spectrum-data")
df = pandas.read_table("F://project//spectrum-data//"+a[0],names=["fc","E"])
df1 = df[df.fc.between(825,830)]
df2 = df[df.fc.between(840,845)]
value1 = df1.values
value2 = df2.values

plt.figure(figsize=(8,4))
plt.scatter(value1[:,0],value1[:,1])
plt.scatter(value2[:,0],value2[:,1])
plt.show()