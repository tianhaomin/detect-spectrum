# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:41:50 2017

@author: Administrator
"""
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
a = os.listdir("F:/project/spectrum-data")
K1 = []
K2 = []
K3 = []
K4 = []
K5 = []
K6 = []
R1 = []
R2 = []
R3 = []
R4 = []
R5 = []
R6 = []
for i in range(800):
    #每次循环先找出间隔50秒的两个频谱数据集
    df1 = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df2 = pandas.read_table("F://project//spectrum-data//"+a[i+50],names=["fc","E"])
    cdmaUpD1 = df1[df1.fc.between(825,830)]#1
    cdmaDownD1 = df1[df1.fc.between(870,875)]#2
    egsmUp1 = df1[df1.fc.between(885,890)]#3
    egsmDown1 = df1[df1.fc.between(930,935)]#4
    wlan1 = df1[df1.fc.between(2400,2405)]#5
    lteY1 = df1[df1.fc.between(1880,1885)]#7
    
    cdmaUpD2 = df2[df2.fc.between(825,830)]#1
    cdmaDownD2 = df2[df2.fc.between(870,875)]#2
    egsmUp2 = df2[df2.fc.between(885,890)]#3
    egsmDown2 = df1[df2.fc.between(930,935)]#4
    wlan2 = df2[df2.fc.between(2400,2405)]#5
    lteY2 = df2[df2.fc.between(1880,1885)]#7
    
    cdmaUpD1 = cdmaUpD1.reset_index()
    cdmaUpD2 = cdmaUpD2.reset_index()
    m1 = cdmaUpD1.E.mean()
    m2 = cdmaUpD2.E.mean()
    for i in range(len(cdmaUpD1)):
        for j in range(len(cdmaUpD2)):
            sum1 = (cdmaUpD1['E'][i]-m1) * (cdmaUpD2['E'][j]-m2)
            sum11 = (cdmaUpD1['E'][i]) * (cdmaUpD2['E'][j])
    k1 = sum1/(len(cdmaUpD1)*len(cdmaUpD2))
    r1 = sum11/(len(cdmaUpD1)*len(cdmaUpD2))
    K1.append(k1)#协方差的提取
    R1.append(r1)
    
    cdmaDownD1 = cdmaDownD1.reset_index()
    cdmaDownD2 = cdmaDownD2.reset_index()
    m3 = cdmaDownD1.E.mean()
    m4 = cdmaDownD2.E.mean()
    for i in range(len(cdmaDownD1)):
        for j in range(len(cdmaDownD2)):
            sum2 = (cdmaDownD1['E'][i]-m3) * (cdmaDownD2['E'][j]-m4)
            sum22 = (cdmaDownD1['E'][i]) * (cdmaDownD2['E'][j])
    k2 = sum2/(len(cdmaDownD1)*len(cdmaDownD2))
    r2 = sum22/(len(cdmaDownD1)*len(cdmaDownD2))
    K2.append(k2)
    R2.append(r2)
    
    egsmUp1 = egsmUp1.reset_index()
    egsmUp2 = egsmUp2.reset_index()
    m5 = egsmUp1.E.mean()
    m6 = egsmUp2.E.mean()
    for i in range(len(egsmUp1)):
        for j in range(len(egsmUp2)):
            sum3 = (egsmUp1['E'][i]-m5) * (egsmUp2['E'][j]-m6)
            sum33 = (egsmUp1['E'][i]) * (egsmUp2['E'][j])
    k3 = sum3/(len(egsmUp1)*len(egsmUp2))
    r3 = sum33/(len(egsmUp1)*len(egsmUp2))
    K3.append(k3)
    R3.append(r3)
    
    egsmDown1 = egsmDown1.reset_index()
    egsmDown2 = egsmDown2.reset_index()
    m7 = egsmDown1.E.mean()
    m8 = egsmDown2.E.mean()
    for i in range(len(egsmDown1)):
        for j in range(len(egsmDown2)):
            sum4 = (egsmDown1['E'][i]-m7) * (egsmDown2['E'][j]-m8)
            sum44 = (egsmDown1['E'][i]) * (egsmDown2['E'][j])
    k4 = sum4/(len(egsmDown1)*len(egsmDown2))
    r4 = sum44/(len(egsmDown1)*len(egsmDown2))
    K4.append(k4)
    R4.append(r4)
    
    wlan1 = wlan1.reset_index()
    wlan2 = wlan2.reset_index()
    m9 = wlan1.E.mean()
    m10 = wlan2.E.mean()
    for i in range(len(wlan1)):
        for j in range(len(wlan2)):
            sum5 = (wlan1['E'][i]-m9) * (wlan2['E'][j]-m10)
            sum55 = (wlan1['E'][i]) * (wlan2['E'][j])
    k5 = sum5/(len(wlan1)*len(wlan2))
    r5 = sum55/(len(wlan1)*len(wlan2))
    K5.append(k5)
    R5.append(r5)
    
    lteY1 = lteY1.reset_index()
    lteY2 = lteY2.reset_index()
    m11 = lteY1.E.mean()
    m12 = lteY2.E.mean()
    for i in range(len(lteY1)):
        for j in range(len(lteY2)):
            sum6 = (lteY1['E'][i]-m11) * (lteY2['E'][j]-m12)
            sum66 = (lteY1['E'][i]) * (lteY2['E'][j])
    k6 = sum6/(len(lteY1)*len(lteY2))
    r6 = sum66/(len(lteY1)*len(lteY2))
    K6.append(k6)
    R6.append(r6)
    
K1 = np.array(K1)
K2 = np.array(K2)
K3 = np.array(K3)
K4 = np.array(K4)
K5 = np.array(K5)
K6 = np.array(K6)
K1 = K1/K1.max()
K2 = K2/K2.max()
K3 = K3/K3.max()
K4 = K4/K4.max()
K5 = K5/K5.max()
K6 = K6/K6.max()
K = np.hstack((K1,K2,K3,K4,K5,K6))

R1 = np.array(R1)
R2 = np.array(R2)
R3 = np.array(R3)
R4 = np.array(R4)
R5 = np.array(R5)
R6 = np.array(R6)
R1 = R1/R1.max()
R2 = R2/R2.max()
R3 = R3/R3.max()
R4 = R4/R4.max()
R5 = R5/R5.max()
R6 = R6/R6.max()
R = np.hstack((R1,R2,R3,R4,R5,R6))


Z = np.stack((R,K)).T


label1 = np.zeros(800).astype(int)
label2 = np.ones(800).astype(int)
label3 = np.ones(800).astype(int)*2
label4 = np.ones(800).astype(int)*3
label5 = np.ones(800).astype(int)*4
label6 = np.ones(800).astype(int)*5
label = np.hstack((label1,label2,label3,label4,label5,label6))


