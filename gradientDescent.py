# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 22:28:55 2017

@author: Administrator
"""
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
#h = a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5
a = os.listdir("F:/project/spectrum-data")
z = []
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df1 = df[df.fc.between(825,835)]#1
    df1['fc'] = df1['fc']/100.0
    z.append(df1)
a0 = 0.0
a1 = 0.0
a2 = 0.0
Y = np.array(z[0]["E"])
X = np.array(z[0]["fc"])
m = float(len(X))
step = 0.000001
e = 1e-2
w = []
def h(x,a0,a1,a2):
    return a0+a1*x+a2*x**2
def costfunc(x,y,a0,a1,a2):
    return np.sum((h(x,a0,a1,a2)-Y).T.dot(h(x,a0,a1,a2)-Y))  
for i in range(1000):
    temp0 = step*(1/m)*np.sum(h(X,a0,a1,a2)-Y)
    temp1 = step*(1/m)*np.sum((h(X,a0,a1,a2)-Y).T.dot(X))
    temp2 = step*(1/m)*np.sum((h(X,a0,a1,a2)-Y).T.dot(X**2))
    a0 = a0-temp0
    a1 = a1-temp1
    a2 = a2-temp2
    #print(costfunc(X,Y,a0,a1,a2))
    print([a0,a1,a2])
    w.append([a0,a1,a2])
    i += 1
w0=[]
w1=[]
w2=[]

for i in w:
    w0.append(i[0])
    w1.append(i[1])
    w2.append(i[2])
    
    
    
lenw = len(w)   
wn = w[lenw-1] 
h = wn[0]+wn[1]*X+wn[2]*(X**2)
costfunc(X,Y,wn[0],wn[1],wn[2])


'''  
Z = costfunc(X,Y,1.0,1.0)
f, (ax1, ax2) = plt.subplots(1, 2)
CS = ax1.contour(X, Y, Z)#画等高线
ax1.plot(w1, w2, 'bo')#把点画出来
ax1.clabel(CS, inline=1, fontsize=10)#给等高线标数值
ax1.set_title('contour')
ax1.set_xlabel('w0')
ax1.set_ylabel('w1')
ax1.legend(('grad', 'Newton'))
ax1.grid(True)
'''
plt.plot(X,Y,'o')
plt.plot(X,h)