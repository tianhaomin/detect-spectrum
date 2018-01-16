# -*- coding: utf-8 -*-
"""
Created on Sun May  7 19:37:42 2017

@author: Administrator
"""

#导出数据
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
a = os.listdir("F:/project/spectrum-data")
cdmaUpD = pandas.DataFrame()
cdmaDownD = pandas.DataFrame()
egsmUp = pandas.DataFrame()
egsmDown = pandas.DataFrame()
wlan = pandas.DataFrame()
lteY = pandas.DataFrame()
df = pandas.DataFrame()
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    cdmaUpD = df[df.fc.between(825,835)]
    cdmaUpD = pandas.concat((cdmaUpD,df))
    cdmaUpD.to_csv("F:/project/groupData/cdmaUpD.csv", index=False)
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    cdmaDownD = df[df.fc.between(870,880)]
    cdmaDownD = pandas.concat((cdmaDownD,df))
    cdmaDownD.to_csv("F:/project/groupData/cdmaDownD.csv", index=False)
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    egsmUp = df[df.fc.between(885,890)]
    egsmUp = pandas.concat((egsmUp,df))
    egsmUp.to_csv("F:/project/groupData/egsmUp.csv", index=False)
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    egsmDown = df[df.fc.between(930,935)]
    egsmDown = pandas.concat((egsmDown,df))
    egsmDown.to_csv("F:/project/groupData/egsmDown.csv", index=False)
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    wlan = df[df.fc.between(2400,2483.5)]
    wlan = pandas.concat((wlan,df))
    wlan.to_csv("F:/project/groupData/wlan.csv", index=False)
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    lteY = df[(df.fc.between(1880,1900))|(df.fc.between(2320,2370))|(df.fc.between(2575,2635))]
    lteY = pandas.concat((lteY,df))
    lteY.to_csv("F:/project/groupData/lteY.csv", index=False)
    
    
#等间隔存储数据取5M
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i])
    cdmaUpD = df[df.iloc[0].between(825,830)]
    cdmaUpD.to_csv("F:/project/groupData/cdmaUpD/%i.csv"%i, index=False)
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i])
    cdmaDownD = df[df.fc.between(870,875)]
    cdmaDownD.to_csv("F:/project/groupData/cdmaDownD/%i.csv"%i, index=False)
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i])
    egsmUp = df[df.fc.between(885,890)]
    egsmUp.to_csv("F:/project/groupData/egsmUp/%i.csv"%i, index=False)
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i])
    egsmDown = df[df.fc.between(930,935)]
    egsmDown.to_csv("F:/project/groupData/egsmDown/%i.csv"%i, index=False)
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i])
    wlan = df[df.fc.between(2400,2405)]
    wlan.to_csv("F:/project/groupData/wlan/%i.csv"%i, index=False)
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i])
    lteY = df[(df.fc.between(1880,1885))]
    lteY.to_csv("F:/project/groupData/lteY/%i.csv"%i, index=False)
