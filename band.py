# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 19:19:08 2017

@author: Administrator
"""
#算带宽
import numpy as np
import pandas
import matplotlib.pyplot as plt
df = pandas.read_table("F://project//spectrum-data//20160921_165221_117.7044_38.9908_1.txt",names=["fc","E"])
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
plt.scatter(cdmaUpD['fc'],cdmaUpD['E'])
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

#draw pic 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
x = z['fc']
y = z['E']
z1 = z['band'] 
ax=plt.subplot(111,projection='3d') #创建一个三维的绘图工程

#将数据点分成三部分画，在颜色上有区分度
ax.scatter(x,y,z1,c='y') #绘制数据点


ax.set_zlabel('Z') #坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()








"""
Created on Thu Mar 16 15:12:54 2017

@author: Administrator
"""
#调用sklearn中的聚类方法
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import pandas
import matplotlib.pyplot as plt
df = pandas.read_table("F://project//spectrum-data//20160921_165222_117.7044_38.9908_2.txt",names=["E","fc"])
D = np.array([df1['fc'],df1['E']])
D1 = D.transpose()#得到了本该有的数据形式
plt.figure(figsize=(12,12))
#N是所有数据集所有训练数据
#feature = [[float(x) for x in row[3:]] for row in data]list简单高效的写法
#接下来是要调用数据
'''
KMeans类的主要参数有：
1) n_clusters: 即我们的k值，一般需要多试一些值以获得较好的聚类效果。
2）max_iter： 最大的迭代次数，一般如果是凸数据集的话可以不管这个值，如果数据集不
是凸的，可能很难收敛，此时可以指定最大的迭代次数让算法可以及时退出循环。
3）n_init：用不同的初始化质心运行算法的次数。由于K-Means是结果受初始值影
响的局部最优的迭代算法，因此需要多跑几次以选择一个较好的聚类效果，默认是10，
一般不需要改。如果你的k值较大，则可以适当增大这个值。
4）init： 即初始值选择的方式，可以为完全随机选择'random',优化过的'k-means++'
或者自己指定初始化的k个质心。一般建议使用默认的'k-means++（这种初始化方法就是将起始点设置的比较远这种初始化的方法要比随机选点要好）'。
5）algorithm：有“auto”, “full” or “elkan”三种选择。"full"就是我们传统的K-Means
算法， “elkan”是我们原理篇讲的elkan K-Means算法。默认的"auto"则会根据数据值是否是
稀疏的，来决定如何选择"full"和“elkan”。一般数据是稠密的，那么就是 “elkan”，
否则就是"full"。一般来说建议直接用默认的"auto"
'''
n_samples = 118801
ypred = KMeans(n_clusters=20,random_state=9).fit_predict(D1)
plt.xlabel("E")
plt.ylabel("fc")
plt.scatter(D1[:, 0], D1[:, 1], c=ypred)
plt.show()
'''
from sklearn import metrics
metrics.calinski_harabaz_score(D1, ypred)  
'''
#对于sklearn的K-means的讲解
'''
X:样本数据集要求的形式是一个数组
K：要聚类的簇的个数
次算法已经自己定义了选择起始点的办法，我们可以自己制定也可以进行算法自己的默认方法自己安排起始值
也可以根据样本数据的稠密程度选择用什么方法聚类
每次循环可以自己指点循环的次数结束也可以自己是自己新的聚类中心不会产生太大差别为标准来停止循环
n_jobs这个参数比较特别就是以消耗内存为代价加速处理=1就是使用一个处理器=2就是两个=-1就是使用所有处理器=-2就是使用少于一个处理器
'''




