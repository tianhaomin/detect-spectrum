# -*- coding: utf-8 -*-
"""
Created on Wed May 24 20:10:10 2017

@author: Administrator
"""

import math
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn
import os
a = os.listdir("F:/project/spectrum-data")
df = pandas.DataFrame()
v1 = []  
v2 = []
v3 = []  
v4 = []  
v5 = []  
v6 = []
d1 = []
d2 = []
d3 = []
d4 = []
d5 = []
d6 = []
def ep(x):
    return pow(10,x)
#cdmaUP
for i in range(850):
    z1 = [] 
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df1 = df[df.fc.between(825,830)]#1
    v1.append(np.array([ep(t) for t in (df1.values[:,1]/10)]).var())
    for i in range(25):
        df11 = df[df.fc.between(825+i*0.25,825+(i+1)*0.25)]
        z1.append(np.array([ep(t) for t in (df11.values[:,1]/10)]).var())
    d1.append(z1)
#cdmaDown
for i in range(850):
    z2 = []
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df1 = df[df.fc.between(870,875)]#1
    v2.append(np.array([ep(t) for t in (df1.values[:,1]/10)]).var())
    for i in range(25):
        df11 = df[df.fc.between(870+i*0.25,870+(i+1)*0.25)]
        z2.append(np.array([ep(t) for t in (df11.values[:,1]/10)]).var())
    d2.append(z2)
#egsmUp
for i in range(850):
    z3 = []
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df1 = df[df.fc.between(885,890)]#1
    v3.append(np.array([ep(t) for t in (df1.values[:,1]/10)]).var())
    for i in range(25):
        df11 = df[df.fc.between(885+i*0.25,885+(i+1)*0.25)]
        z3.append(np.array([ep(t) for t in (df11.values[:,1]/10)]).var())
    d3.append(z3)
#egsmDown
for i in range(850):
    z4 = []
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df1 = df[df.fc.between(930,935)]#1
    v4.append(np.array([ep(t) for t in (df1.values[:,1]/10)]).var())
    for i in range(25):
        df11 = df[df.fc.between(930+i*0.25,930+(i+1)*0.25)]
        z4.append(np.array([ep(t) for t in (df11.values[:,1]/10)]).var())
    d4.append(z4)
#wlan
for i in range(850):
    z5 = []
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df1 = df[df.fc.between(2400,2405)]#1
    v5.append(np.array([ep(t) for t in (df1.values[:,1]/10)]).var())
    for i in range(25):
        df11 = df[df.fc.between(2400+i*0.25,2400+(i+1)*0.25)]
        z5.append(np.array([ep(t) for t in (df11.values[:,1]/10)]).var())
    d5.append(z5)
#lteY
for i in range(850):
    z6 = []
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df1 = df[df.fc.between(1880,1885)]#1
    v6.append(np.array([ep(t) for t in (df1.values[:,1]/10)]).var())
    for i in range(25):
        df11 = df[df.fc.between(1880+i*0.25,1880+(i+1)*0.25)]
        z6.append(np.array([ep(t) for t in (df11.values[:,1]/10)]).var())
    d6.append(z6)    

V1 = v1[0]
for i in range(1,len(v1)):
    V1 = np.hstack((V1,v1[i]))   
 
V2 = v2[0]
for i in range(1,len(v2)):
    V2 = np.hstack((V2,v2[i])) 

V3 = v3[0]
for i in range(1,len(v3)):
    V3 = np.hstack((V3,v3[i])) 
    
V4 = v4[0]
for i in range(1,len(v4)):
    V4 = np.hstack((V4,v4[i])) 
    
V5 = v5[0]
for i in range(1,len(v5)):
    V5 = np.hstack((V5,v5[i])) 
    
V6 = v6[0]
for i in range(1,len(v6)):
    V6 = np.hstack((V6,v6[i]))
V = np.hstack((V1,V2,V3,V4,V5,V6)).T



D1 = np.array(d1)
D2 = np.array(d2)
D3 = np.array(d3)
D4 = np.array(d4)
D5 = np.array(d5)
D6 = np.array(d6)
D = np.stack((D1,D2,D3,D4,D5,D6))
D.shape=(5100,25)

    
#绘制方差图像 
l = np.arange(1,851,1)
plt.figure(figsize=(8,4))
plt.plot(l,v1,color='r',linestyle='-',marker='.',label=u"cdmaUpVar")
plt.plot(l,v2,color='g',linestyle='-',marker='.',label=u"cdmaDownVar")
plt.plot(l,v3,color='b',linestyle='-',marker='.',label=u"egsmUpVar")
plt.plot(l,v4,color='y',linestyle='-',marker='.',label=u"egsmDownVar")
plt.plot(l,v5,color='#00FFFF',linestyle='-',marker='.',label=u"wlanVar")
plt.plot(l,v6,color='#A9A9A9',linestyle='-',marker='.',label=u"lteVar")
plt.legend()
plt.show()
plt.figure()
#绘制方差分布的图像
e = np.arange(1,26,1)
plt.figure(figsize=(8,4))
plt.plot(e,d1[0],color='r',linestyle='-',marker='.',label=u"cdmaUp")
plt.plot(e,d2[0],color='g',linestyle='-',marker='.',label=u"cdmaDown")
plt.plot(e,d3[0],color='b',linestyle='-',marker='.',label=u"egsmUp")
plt.plot(e,d4[0],color='y',linestyle='-',marker='.',label=u"egsmDown")
plt.plot(e,d5[0],color='#00FFFF',linestyle='-',marker='.',label=u"wlan")
plt.plot(e,d6[0],color='#A9A9A9',linestyle='-',marker='.',label=u"lte")
#plt.ylim(0,7)
plt.legend()
plt.show()
#keras对方差分布进行分类

