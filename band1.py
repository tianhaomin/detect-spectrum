# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:39:16 2017

@author: Administrator
"""

import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
a = os.listdir("F:/project/spectrum-data")
from mpl_toolkits.mplot3d import Axes3D
cdmaUpD = pandas.DataFrame()
cdmaDownD = pandas.DataFrame()
egsmUp = pandas.DataFrame()
egsmDown = pandas.DataFrame()
wlan = pandas.DataFrame()
lteY = pandas.DataFrame()
df = pandas.DataFrame()

for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    cdmaUpD = df[df.fc.between(825,835)]#1
    cdmaUpD["name"] = "1"
    cdmaDownD = df[df.fc.between(870,880)]#2
    cdmaDownD["name"] = "2"
    egsmUp = df[df.fc.between(885,890)]#3
    egsmUp["name"] = "3"
    egsmDown = df[df.fc.between(930,935)]#4
    egsmDown["name"] = "4"
    wlan = df[df.fc.between(2400,2483.5)]#5
    wlan["name"] = "5"
    wd = df[df.fc.between(3400,3600)]#6
    wd["name"] = "6"
    lteY = df[(df.fc.between(1880,1900))|(df.fc.between(2320,2370))|(df.fc.between(2575,2635))]#7
    lteY["name"] = "7"
    df1 = pandas.concat([cdmaUpD,cdmaDownD,egsmUp,egsmDown,wlan,lteY])
    #计算带宽
   #plt.scatter(cdmaUpD['fc'],cdmaUpD['E'])
    #10lgE->10lgE/2下降了3db,逐个读点可以找到下降3db的点但是不会找到频率的间隔
    #cdmaUpD带宽
    sum1 = 0
    num1 = 0
    cdmaUpD = cdmaUpD.reset_index()
    for i in range(len(cdmaUpD)-1):
        det = cdmaUpD['E'][i]-cdmaUpD['E'][i+1]
        if det>3:
            sum1 += (cdmaUpD['fc'][i]-cdmaUpD['fc'][i+1])
            num1 += 1
    band1 = sum1/num1
    #cdmaDownD带宽
    sum2 = 0
    num2 = 0
    cdmaDownD = cdmaDownD.reset_index()
    for i in range(len(cdmaDownD)-1):
        det = cdmaDownD['E'][i]-cdmaDownD['E'][i+1]
        if det>3:
            sum2 += (cdmaDownD['fc'][i]-cdmaDownD['fc'][i+1])
            num2 += 1
    band2 = sum2/num2       
    #egsmUp band
    sum3 = 0
    num3 = 0
    egsmUp = egsmUp.reset_index()
    for i in range(len(egsmUp)-1):
        det = egsmUp['E'][i]-egsmUp['E'][i+1]
        if det>3:
            sum3 += (egsmUp['fc'][i]-egsmUp['fc'][i+1])
            num3 += 1
    band3 = sum3/num3
    #egsmDown band   
    sum4 = 0
    num4 = 0
    egsmDown = egsmDown.reset_index()
    for i in range(len(egsmDown)-1):
        det = egsmDown['E'][i]-egsmDown['E'][i+1]
        if det>3:
            sum4 += (egsmDown['fc'][i]-egsmDown['fc'][i+1])
            num4 += 1
    band4 = sum4/num4
    #wlan band
    sum5 = 0
    num5 = 0
    wlan = wlan.reset_index()
    for i in range(len(wlan)-1):
        det = wlan['E'][i]-wlan['E'][i+1]
        if det>3:
            sum5 += (wlan['fc'][i]-wlan['fc'][i+1])
            num5 += 1
    band5 = sum5/num5
    #lte band
    sum6 = 0
    num6 = 0
    lteY = lteY.reset_index()
    for i in range(len(lteY)-1):
        det = lteY['E'][i]-lteY['E'][i+1]
        if det>3:
            sum6 += (lteY['fc'][i]-lteY['fc'][i+1])
            num6 += 1
    band6 = sum6/num6
    #将
    df['name'] = 'else'
    cdmaUpD['band'] = band1
    cdmaDownD['band'] = band2
    egsmUp['band'] = band3
    egsmDown['band'] = band4
    wlan['band'] = band5
    lteY['band'] = band6
    z = pandas.concat([cdmaUpD,cdmaDownD,egsmUp,egsmDown,wlan,lteY])
    x = z['fc']
    y = z['E']
    z1 = z['band'] 
    ax = plt.subplot(111, projection='3d')
    #将数据点分成三部分画，在颜色上有区分度
    ax.scatter(x,y,z1,c='y') #绘制数据点
    ax.set_zlabel('band') #坐标轴
    ax.set_ylabel('fc')
    ax.set_xlabel('E')
    #plt.savefig("F:/project/operatordata/pics/clusterband/%i.png"%i)
    plt.show()
