# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 15:57:03 2017

@author: Administrator
"""

import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
a = os.listdir("F:/project/spectrum-data")
Band1 = []
Band2 = []
Band3 = []
Band4 = []
Band5 = []
Band6 = []

for i in range(len(a)):
    
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    cdmaUpD = df[df.fc.between(825,835)]#1
    cdmaDownD = df[df.fc.between(870,880)]#2
    egsmUp = df[df.fc.between(885,890)]#3
    egsmDown = df[df.fc.between(930,935)]#4
    wlan = df[df.fc.between(2400,2483.5)]#5
    lteY = df[(df.fc.between(1880,1900))|(df.fc.between(2320,2370))|(df.fc.between(2575,2635))]#7
    
    sum1 = 0
    num1 = 0
    cdmaUpD = cdmaUpD.reset_index()
    for i in range(len(cdmaUpD)-1):
        det = cdmaUpD['E'][i]-cdmaUpD['E'][i+1]
        if det>3:
            sum1 += (cdmaUpD['fc'][i]-cdmaUpD['fc'][i+1])
            num1 += 1
    band1 = sum1/num1
    Band1.append(band1)
    
    
    sum2 = 0
    num2 = 0
    cdmaDownD = cdmaDownD.reset_index()
    for i in range(len(cdmaDownD)-1):
        det = cdmaDownD['E'][i]-cdmaDownD['E'][i+1]
        if det>3:
            sum2 += (cdmaDownD['fc'][i]-cdmaDownD['fc'][i+1])
            num2 += 1
    band2 = sum2/num2 
    Band2.append(band2)
    
    sum3 = 0
    num3 = 0
    egsmUp = egsmUp.reset_index()
    for i in range(len(egsmUp)-1):
        det = egsmUp['E'][i]-egsmUp['E'][i+1]
        if det>3:
            sum3 += (egsmUp['fc'][i]-egsmUp['fc'][i+1])
            num3 += 1
    band3 = sum3/num3
    Band3.append(band3)
    
    sum4 = 0
    num4 = 0
    egsmDown = egsmDown.reset_index()
    for i in range(len(egsmDown)-1):
        det = egsmDown['E'][i]-egsmDown['E'][i+1]
        if det>3:
            sum4 += (egsmDown['fc'][i]-egsmDown['fc'][i+1])
            num4 += 1
    band4 = sum4/num4
    Band4.append(band4)
    
    sum5 = 0
    num5 = 0
    wlan = wlan.reset_index()
    for i in range(len(wlan)-1):
        det = wlan['E'][i]-wlan['E'][i+1]
        if det>3:
            sum5 += (wlan['fc'][i]-wlan['fc'][i+1])
            num5 += 1
    band5 = sum5/num5
    Band5.append(band5)
    
    sum6 = 0
    num6 = 0
    lteY = lteY.reset_index()
    for i in range(len(lteY)-1):
        det = lteY['E'][i]-lteY['E'][i+1]
        if det>3:
            sum6 += (lteY['fc'][i]-lteY['fc'][i+1])
            num6 += 1
    band6 = sum6/num6
    Band6.append(band6)
    
    