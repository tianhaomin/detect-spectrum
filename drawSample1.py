# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:52:46 2017

@author: Administrator
"""
#把样本点画出来(样本1)
from pandas import read_table
import matplotlib.pyplot as plt
df = read_table("F://project//spectrum-data//20160921_165221_117.7044_38.9908_1.txt",names=["fc","E"])
plt.plot(text1['fc'],text1['E'])#把各个点连接起来
cdmaDownD = df[df.fc.between(870,880)]#2    plt.figure(figsize=(8,4))
    plt.scatter(cdmaDownD['fc'],cdmaDownD['E'])
    plt.xlabel('fc')
    plt.ylabel('E')
plt.xlim(-60,60)
plt.ylim(0, 3000)
x = []
y = []
#plt.plot(text1['fc'],text1['E'])
#text1['E'][0] 可以对panadas中的值进行存取

for i in range(len(text1)):
    x.append(text1['fc'][i])
    y.append(text1['E'][i])
    if i % 100 == 0:
        plt.figure(figsize=(8,4))
        plt.plot(x,y)
        plt.xlabel('f')
        plt.ylabel('E')
        plt.savefig("F:/project/operatordata/picture/sample2/sample2Figure2/%i.png"%i)
        plt.show()
        x = []
        y = []
        
