# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:34:00 2017

@author: Administrator
"""
#阉割后的kmeans需要进行循环，自己写的简单的单一聚类
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
plt.figure(figsize=(12,12))
text1 = pandas.read_table("F://project//spectrum-data//20160921_165221_117.7044_38.9908_1.txt",names=["E","fc"])
D = np.array([text1['fc'],text1['E']])
N = []
#N是所有数据集所有训练数据
for i in range(len(text1)):
    a = [D[0][i],D[1][i]]
    N.append(a)#数据集已经准备好
def calc_e_squire(a, b):
    return (a[0]- b[0]) ** 2 + (a[1] - b[1]) **2

#define k k_value
k1 = [-8.3,30.0]
k2 = [-14.3,500]
k3 = [-2.9,1517.5]
k4 = [-8.3,2997.975]

#defint k cluster
sse_k1 = []
sse_k2 = []
sse_k3 = []
sse_k4 = []
while True:
    sse_k1 = []
    sse_k2 = []
    sse_k3 = []
    sse_k4 = []
    for i in range(len(N)):
        e_squire1 = calc_e_squire(k1, [N[i][0], N[i][1]])
        e_squire2 = calc_e_squire(k2, [N[i][0], N[i][1]])
        e_squire3 = calc_e_squire(k3, [N[i][0], N[i][1]])
        e_squire4 = calc_e_squire(k4, [N[i][0], N[i][1]])
        minc = min(e_squire1,e_squire2,e_squire3,e_squire4,)
        if (e_squire1 == minc):
            sse_k1.append(i) #我们添加的是list的标号
        elif (e_squire2 == minc):
            sse_k2.append(i)
        elif (e_squire3 == minc):
            sse_k3.append(i)
        else:
            sse_k4.append(i)

    #change k_value
    k1_x = sum([N[i][0] for i in sse_k1]) / len(sse_k1)
    k1_y = sum([N[i][1] for i in sse_k1]) / len(sse_k1)

    k2_x = sum([N[i][0] for i in sse_k2]) / len(sse_k2)
    k2_y = sum([N[i][1] for i in sse_k2]) / len(sse_k2)
    
    k3_x = sum([N[i][0] for i in sse_k3]) / len(sse_k3)
    k3_y = sum([N[i][1] for i in sse_k3]) / len(sse_k3)

    k4_x = sum([N[i][0] for i in sse_k4]) / len(sse_k4)
    k4_y = sum([N[i][1] for i in sse_k4]) / len(sse_k4)

    if k1 != [k1_x, k1_y] or k2 != [k2_x, k2_y] or k3 != [k3_x, k3_y] or k4 != [k4_x, k4_y]:
        k1 = [k1_x, k1_y]
        k2 = [k2_x, k2_y]
        k3 = [k3_x, k3_y]
        k4 = [k4_x, k4_y]
    else:
        break

kv1_x = [N[i][0] for i in sse_k1]
kv1_y = [N[i][1] for i in sse_k1]

kv2_x = [N[i][0] for i in sse_k2]
kv2_y = [N[i][1] for i in sse_k2]

kv3_x = [N[i][0] for i in sse_k3]
kv3_y = [N[i][1] for i in sse_k3]

kv4_x = [N[i][0] for i in sse_k4]
kv4_y = [N[i][1] for i in sse_k4]

plt.plot(kv1_x, kv1_y, 'o')
plt.plot(kv2_x, kv2_y, 'or')
plt.plot(kv3_x, kv3_y, 'o')
plt.plot(kv4_x, kv4_y, 'or')

plt.xlim(-20, 0.000630)
plt.ylim(-0.000630, 3000)
plt.show()