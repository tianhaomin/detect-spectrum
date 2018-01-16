w# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:31:31 2017

@author: Administrator
"""

import numpy as np  
import pandas
import matplotlib.pyplot as plt
import os
z = []
cdmaUpD = pandas.DataFrame()
a = os.listdir("F:/project/spectrum-data")
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df1 = df[df.fc.between(825,825.5)]#1
    z.append(df1)
l = len(z)
M = []
for i in range(l):
    length = len(z[i])
    summ = z[i]["E"].sum()
    mean = summ/length
    M.append(mean)
lamda=np.mean(M)
mc = []
for i in range(l-1):
    large = len(z[i].loc[z[i]["E"]>lamda])
    small = len(z[i].loc[z[i]["E"]<=lamda])
    p0 = small/float(len(z[i]))
    p1 = 1-p0
    large1 = len(z[i+1].loc[z[i+1]["E"]>lamda])
    small1 = len(z[i+1].loc[z[i+1]["E"]<=lamda])
    P0 = small1/float(len(z[i+1]))
    P1 = 1 - P0
    p00 = p0*P0
    p01 = p0*P1
    p10 = p1*P0
    p11 = p1*P1
    mc.append(np.array([[p00,p01],[p10,p11]]))
summ1 = 0.0
summ2 = 0.0
summ3 = 0.0
summ4 = 0.0
Mar = np.array([[0.0,0.0],[0.0,0.0]])
for i in mc:
    summ1 += i[0][0]
    summ2 += i[0][1]
    summ3 += i[1][0]
    summ4 += i[1][1]    
    Mar[0][0] = summ1/len(mc)
    Mar[0][1] = summ2/len(mc)
    Mar[1][0] = summ3/len(mc)
    Mar[1][1] = summ4/len(mc)
    