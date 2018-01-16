# -*- coding: utf-8 -*-
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
text1 = pandas.read_table("F://project//spectrum-data//20160921_165222_117.7044_38.9908_2.txt",names=["E","fc"])
D = np.array([text1['E'],text1['fc']])
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
plt.xlim(0,3000)
plt.ylim(-20,0)
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




