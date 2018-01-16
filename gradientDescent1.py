# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 12:02:30 2017

@author: Administrator
"""
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

a = os.listdir("F:/project/spectrum-data")
z = []
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    cdmaUpD = df[df.fc.between(825,835)]
    cdmaUpD['fc'] = cdmaUpD['fc']/100.0
    z.append(cdmaUpD)

m = len(cdmaUpD)
a1 = np.ones(m)
A1 = np.array([a1,cdmaUpD['fc']])
A = A1.T/1000000
B = np.array([cdmaUpD['E']])
B = B.T
R=A.T.dot(A)
P=A.T.dot(B)
X, Y = np.meshgrid(np.linspace(-5, 10, 100), np.linspace(-5, 10, 100))#曲
def cost_func(X,Y):
    return R[0,0]*(X**2)+R[1,1]*(Y**2)+(R[0,1]+R[1,0])*X*Y-2*P[0,0]*X-2*P[1,0]*Y
z = cost_func(X,Y)
w=np.array([[1.0],[1.0]])
for i in range(100):
    w_t = w[:,-1]
    w=np.hstack((w,w_t-0.01*2*(R.dot(w_t)-P)))
w = np.array(w)
L=cost_func(w[0,:], w[1,:])



f, (ax1, ax2) = plt.subplots(1, 2)
CS = ax1.contour(X, Y, z)#画等高线
ax1.plot(w[0,:], w[1,:], 'bo')#把点画出来

ax1.clabel(CS, inline=1, fontsize=10)#给等高线标数值
ax1.set_title('contour')
ax1.set_xlabel('w0')
ax1.set_ylabel('w1')
ax1.legend(('grad', 'Newton'))
ax1.grid(True)


ax2.plot(range(201),L,'b')
ax2.set_title('cost learn curve')
ax2.set_ylabel('cost')
ax2.legend(('grad'))
ax2.grid(True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y, z, rstride=4, cstride=4, color='b')
ax.plot(w[0,:],w[1,:],L,'ro')
ax.legend(('grad'))

plt.show()