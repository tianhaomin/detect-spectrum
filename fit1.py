# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:38:13 2017

@author: Administrator
"""
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn
import os
a = os.listdir("F:/project/spectrum-data")
lteY = pandas.DataFrame()
df = pandas.DataFrame()
z=[]
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df1 = df[df.fc.between(825,835)]#1
    df1['fc'] = df1['fc']
    z.append(df1)
x = z[0]['fc']  
x = np.array(x)  
m = []
for i in range(5):
    m.append(x**(i))  
A = np.array(m).T
y1 = z[0]['E']
y = np.array(y1)
b = y.reshape(y.shape[0],1)
def projection(A,b):
    ####
    # return A*inv(AT*A)*AT*b
    ####
    AA = A.T.dot(A)
    w=np.linalg.inv(AA).dot(A.T).dot(b)
    print (w)
    return A.dot(w)

yw = projection(A,b)
yw.shape = (yw.shape[0],)
plt.plot(x,y,color='m',linestyle='',marker='o',label=u"已知数据点")
plt.plot(x,yw,color='r',linestyle='-',marker='.',label=u"拟合曲线")

plt.legend()
plt.show()