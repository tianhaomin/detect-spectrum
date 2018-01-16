# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:23:38 2017

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

def ep(x):
    return pow(10,x)
mean1 = []
k1 = []
for i in range(850):
    z1=[]
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df1 = df[df.fc.between(825,830)]#1
    k1.append(np.array([ep(t) for t in (df1.values[:,1]/10)]).mean())
    for i in range(20):
        df11 = df[df.fc.between(825+i*0.25,825+(i+1)*0.25)]
        z1.append(np.array([ep(t) for t in (df11.values[:,1]/10)]).mean())
    mean1.append(z1)
    
 
mean2=[]
k2 = []
for i in range(850):
    z2=[]
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df2 = df[df.fc.between(870,875)]#1
    k2.append(np.array([ep(t) for t in (df2.values[:,1]/10)]).mean())
    for i in range(20):
        df22 = df[df.fc.between(870+i*0.25,870+(i+1)*0.25)]
        z2.append(np.array([ep(t) for t in (df22.values[:,1]/10)]).mean())
    mean2.append(z2)    
        

mean3 = []
k3 = []
for i in range(850):
    z3 = []
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df3 = df[df.fc.between(885,890)]#1
    k3.append(np.array([ep(t) for t in (df3.values[:,1]/10)]).mean())
    for i in range(20):
        df33 = df[df.fc.between(885+i*0.25,885+(i+1)*0.25)]
        z3.append(np.array([ep(t) for t in (df33.values[:,1]/10)]).mean())
    mean3.append(z3)

mean4 = []
k4 = []
for i in range(850):
    z4=[]
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df4 = df[df.fc.between(930,935)]#1
    k4.append(np.array([ep(t) for t in (df4.values[:,1]/10)]).mean())
    for i in range(20):
        df44 = df[df.fc.between(930+i*0.25,930+(i+1)*0.25)]
        z4.append(np.array([ep(t) for t in (df44.values[:,1]/10)]).mean())
    mean4.append(z4)    
        

mean5 = []
k5 = []
for i in range(850):
    z5 = []
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df5 = df[df.fc.between(2400,2405)]#1
    k5.append(np.array([ep(t) for t in (df5.values[:,1]/10)]).mean())
    for i in range(20):
        df55 = df[df.fc.between(2400+i*0.25,2400+(i+1)*0.25)]
        z5.append(np.array([ep(t) for t in (df55.values[:,1]/10)]).mean())
    mean5.append(z5)
    
    
mean6 = []
k6 = []
for i in range(850):
    z6 = []
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df6 = df[df.fc.between(1880,1885)]#6 #1
    k6.append(np.array([ep(t) for t in (df6.values[:,1]/10)]).mean())
    for i in range(20):
        df66 = df[df.fc.between(1880+i*0.25,1880+(i+1)*0.25)]
        z6.append(np.array([ep(t) for t in (df66.values[:,1]/10)]).mean())
    mean6.append(z6)
 
#对均值进行keras封装
K1 = k1[0]
for i in range(1,len(k1)):
    K1 = np.hstack((K1,k1[i]))   
 
K2 = k2[0]
for i in range(1,len(k2)):
    K2 = np.hstack((K2,k2[i])) 

K3 = k3[0]
for i in range(1,len(k3)):
    K3 = np.hstack((K3,k3[i])) 
    
K4 = k4[0]
for i in range(1,len(k4)):
    K4 = np.hstack((K4,k4[i])) 
    
K5 = k5[0]
for i in range(1,len(k5)):
    K5 = np.hstack((K5,k5[i])) 
    
K6 = k6[0]
for i in range(1,len(k6)):
    K6 = np.hstack((K6,k6[i]))
K = np.hstack((K1,K2,K3,K4,K5,K6)).T       


#画出频谱等频带能量均值
e = np.arange(1,21,1)
plt.figure(figsize=(12,12))
plt.plot(e,mean1[1],color='r',linestyle='-',marker='.',label=u"cdmaUp1")
plt.plot(e,mean1[678],color='g',linestyle='-',marker='.',label=u"cdmaUp678")
plt.plot(e,mean1[100],color='b',linestyle='-',marker='.',label=u"cdmaUp100")
plt.plot(e,mean1[45],color='y',linestyle='-',marker='.',label=u"cdmaUp45")
#plt.plot(e,z5,color='#00FFFF',linestyle='-',marker='.',label=u"wlan")
#plt.plot(e,z6,color='#A9A9A9',linestyle='-',marker='.',label=u"lte")
#plt.ylim(-15,30)
plt.legend()
plt.show()
plt.figure()


plt.figure(figsize=(12,12))
plt.plot(e,mean2[1],color='r',linestyle='-',marker='.',label=u"cdmaDown1")
plt.plot(e,mean2[678],color='g',linestyle='-',marker='.',label=u"cdmaDown678")
plt.plot(e,mean2[100],color='b',linestyle='-',marker='.',label=u"cdmaDown100")
plt.plot(e,mean2[45],color='y',linestyle='-',marker='.',label=u"cdmaDown45")
plt.plot(e,z5,color='#00FFFF',linestyle='-',marker='.',label=u"wlan")
plt.plot(e,z6,color='#A9A9A9',linestyle='-',marker='.',label=u"lte")
#plt.ylim(-15,30)
plt.legend()
plt.show()
plt.figure()


plt.figure(figsize=(12,12))
plt.plot(e,mean3[1],color='r',linestyle='-',marker='.',label=u"egsmUP1")
plt.plot(e,mean3[678],color='g',linestyle='-',marker='.',label=u"egsmUP1")
plt.plot(e,mean3[100],color='b',linestyle='-',marker='.',label=u"egsmUP1")
plt.plot(e,mean3[45],color='y',linestyle='-',marker='.',label=u"egsmUP1")
plt.plot(e,z5,color='#00FFFF',linestyle='-',marker='.',label=u"wlan")
plt.plot(e,z6,color='#A9A9A9',linestyle='-',marker='.',label=u"lte")
#plt.ylim(-15,30)
plt.legend()
plt.show()
plt.figure()


plt.figure(figsize=(12,12))
plt.plot(e,mean4[1],color='r',linestyle='-',marker='.',label=u"egsmDown1")
plt.plot(e,mean4[678],color='g',linestyle='-',marker='.',label=u"egsmDown678")
plt.plot(e,mean4[100],color='b',linestyle='-',marker='.',label=u"egsmDown100")
plt.plot(e,mean4[45],color='y',linestyle='-',marker='.',label=u"egsmDown45")
#plt.plot(e,z5,color='#00FFFF',linestyle='-',marker='.',label=u"wlan")
#plt.plot(e,z6,color='#A9A9A9',linestyle='-',marker='.',label=u"lte")
plt.ylim(-15,30)
plt.legend()
plt.show()
plt.figure()


plt.figure(figsize=(12,12))
plt.plot(e,mean5[1],color='r',linestyle='-',marker='.',label=u"wlan1")
plt.plot(e,mean5[678],color='g',linestyle='-',marker='.',label=u"wlan678")
plt.plot(e,mean5[100],color='b',linestyle='-',marker='.',label=u"wlan100")
plt.plot(e,mean5[45],color='y',linestyle='-',marker='.',label=u"wlan45")
#plt.plot(e,z5,color='#00FFFF',linestyle='-',marker='.',label=u"wlan")
#plt.plot(e,z6,color='#A9A9A9',linestyle='-',marker='.',label=u"lte")
plt.ylim(-15,30)
plt.legend()
plt.show()
plt.figure()



plt.figure(figsize=(12,12))
plt.plot(e,mean6[1],color='r',linestyle='-',marker='.',label=u"lte1")
plt.plot(e,mean6[678],color='g',linestyle='-',marker='.',label=u"lte678")
plt.plot(e,mean6[100],color='b',linestyle='-',marker='.',label=u"lte100")
plt.plot(e,mean6[45],color='y',linestyle='-',marker='.',label=u"lte45")
#plt.plot(e,z5,color='#00FFFF',linestyle='-',marker='.',label=u"wlan")
#plt.plot(e,z6,color='#A9A9A9',linestyle='-',marker='.',label=u"lte")
plt.ylim(-15,30)
plt.legend()
plt.show()
plt.figure()



plt.figure(figsize=(8,4))
plt.plot(e,mean1[400],color='r',linestyle='-',marker='.',label=u"cdmaUp")
#plt.plot(e,mean2[400],color='g',linestyle='-',marker='.',label=u"cdmaDown")
plt.plot(e,mean3[400],color='b',linestyle='-',marker='.',label=u"egsmUp")
#plt.plot(e,mean4[400],color='y',linestyle='-',marker='.',label=u"egsmDown")
plt.plot(e,mean5[400],color='#00FFFF',linestyle='-',marker='.',label=u"wlan")
plt.plot(e,mean6[400],color='#A9A9A9',linestyle='-',marker='.',label=u"lte")
#plt.plot(e,z5,color='#00FFFF',linestyle='-',marker='.',label=u"wlan")
#plt.plot(e,z6,color='#A9A9A9',linestyle='-',marker='.',label=u"lte")
#plt.ylim(0,7)
plt.legend()
plt.show()
plt.figure()



q = np.arange(1,851,1)
plt.figure(figsize=(12,12))
plt.plot(q,k1,color='r',linestyle='-',marker='.',label=u"cdmaUp")
#plt.plot(q,k2,color='g',linestyle='-',marker='.',label=u"cdmaDown")
plt.plot(q,k3,color='b',linestyle='-',marker='.',label=u"egsmUp")
#plt.plot(q,k4,color='y',linestyle='-',marker='.',label=u"egsmDown")
plt.plot(q,k5,color='#00FFFF',linestyle='-',marker='.',label=u"wlan")
plt.plot(q,k6,color='#A9A9A9',linestyle='-',marker='.',label=u"lte")
#plt.plot(e,z5,color='#00FFFF',linestyle='-',marker='.',label=u"wlan")
#plt.plot(e,z6,color='#A9A9A9',linestyle='-',marker='.',label=u"lte")
#plt.ylim(0,10)
plt.legend()
plt.show()
plt.figure()

#把频谱均值分布作为输入进行keras分类
mean1 = np.array(mean1)
mean2 = np.array(mean2)
mean3 = np.array(mean3)
mean4 = np.array(mean4)
mean5 = np.array(mean5)
mean6 = np.array(mean6)
k = np.stack((mean1,mean2,mean3,mean4,mean5,mean6))
k.shape=(5100,20)