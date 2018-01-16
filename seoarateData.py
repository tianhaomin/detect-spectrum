# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:31:34 2017

@author: Administrator
"""
#数据处理
import numpy as np
import pandas
import matplotlib.pyplot as plt
text1 = pandas.read_table("F://project//spectrum-data//20160921_165221_117.7044_38.9908_1.txt",names=["E","fc"])
'''#划分频谱业务
text1['fc'] = abs(text1['fc'])
cdmaUpD = text1[text1.fc.between(8.25,8.35)]#1
cdmaDownD = text1[text1.fc.between(8.70,8.80)]#2
egsmUp = text1[text1.fc.between(8.85,8.90)]#3
egsmDown = text1[text1.fc.between(9.3,9.35)]#4
wlan = text1[text1.fc.between(24.0,24.835)]#5
wd = text1[text1.fc.between(34,36)]#6
lteY = text1[(text1.fc.between(18.8,19))|(text1.fc.between(23.2,23.7))|(text1.fc.between(25.75,26.35))]#7
figure1 = plt.figure(figsize=(12,12))
plt.scatter(cdmaUpD['fc'],cdmaUpD['E'])
#去10组样本'''
import os
a = os.listdir("F:/project/spectrum-data")
df1 = pandas.read_table("F://project//spectrum-data//20160921_165221_117.7044_38.9908_1.txt",names=["fc","E"])
df=df1
for i in range(1,10):
    df1 = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    df = pandas.concat([df,df1])
#df['E'] = abs(df['E'])
df["name"] = "else"
df = df.reset_index()#重置索引
#业务分类
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
else1 = df[(df.fc.between(0,825))|(df.fc.between(835,870))|(df.fc.between(880,885))|(df.fc.between(890,930))]
else2 = df[(df.fc.between(935,1880))|(df.fc.between(1900,2320))|(df.fc.between(2370,2400))|(df.fc.between(2483.5,2575))]
else3 = df[(df.fc.between(2635,3400))|(df.fc.between(3600,8000))]
cdmaUpD = cdmaUpD.reset_index()
cdmaDownD = cdmaDownD.reset_index()
egsmUp = egsmUp.reset_index()
egsmDown = egsmDown.reset_index()
wlan = wlan.reset_index()
wd = wd.reset_index()
lteY = lteY.reset_index()
#最终总的数据
df2 = pandas.concat([cdmaUpD,cdmaDownD,egsmUp,egsmDown,wlan,wd,lteY,else1,else2,else3])
df2.reset_index()
del df2["level_0"]
del df2["index"]
pcdmau = 11972/1188010
pcdmad = 23364/1188010
pegsmu = 11319/1188010
pegsmd = 11032/1188010
plte = 4352/1188010
pwlan = 1371/1188010
pwd = 121/1188010
plt.figure(figsize=(20,20))
plt.plot(df['fc'],df['E'])
plt.limx(6,9)
'''
说明：
df：所有划分的数据和没有划分的数据融合
df2:划分出来的所有数据的融合

'''
'''
c1 = df.fc.between(8.25,8.35)
for j in range(len(c1)):
    if c1[j] == True:
        df.name[j] = "1"
            
c2 = df.fc.between(8.7,8.8)
for j in range(len(c2)):
    if c2[j] == True:
        df.name[j] = "2"

c3 = df.fc.between(8.85,8.90)
for j in range(len(c3)):
    if c3[j] == True:
        df.name[j] = "3"

c4 = i.fc.between(9.3,9.35)
for j in range(len(c4)):
    if c4[j] == True:
        df.name[j] = "4"

c5 = i.fc.between(24.0,24.835)
for j in range(len(c5)):
    if c5[j] == True:
        df.name[j] = "5"

c6 = i.fc.between(34.0,36.0)
for j in range(len(c6)):
    if c6[j] == True:
        df.name[j] = "6"

c7 = i.fc.between(18.8,19)
for j in range(len(c7)):
    if c7[j] == True:
        df.name[j] = "7"

c8 = i.fc.between(23.3,23.7)
for j in range(len(c8)):
    if c8[j] == True:
        df.name[j] = "7"

c9 = i.fc.between(25.75,26.35)
for j in range(len(c9)):
    if c9[j] == True:
        df.name[j] = "7"
'''      
from pandas import Series
x = Series(["1"]*11)

df1[df1.fc.between(25.75,26.35)]['fc'] = x
