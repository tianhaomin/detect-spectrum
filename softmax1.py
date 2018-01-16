# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 17:34:09 2017

@author: Administrator
"""

import numpy as np  
import matplotlib.pylab as plt  
import copy  
from scipy.linalg import norm  
from math import pow  
from scipy.optimize import fminbound,minimize  
import random  
def _dot(a, b):  
    # 表示点乘
    mat_dot = np.dot(a, b)  
    # exp表示e的多次次幂
    return np.exp(mat_dot)  

def condProb(theta, thetai, xi):  
    # 计算样本i的输出
    numerator = _dot(thetai, xi.transpose())  

    # 计算所有样本的输出求和
    denominator = _dot(theta, xi.transpose())  
    denominator = np.sum(denominator, axis=0)  
    p = numerator / denominator  
    return p  

'''  下面就是求得代价函数 '''
def costFunc(alfa, *args):  
    i = args[2]  
    original_thetai = args[0]  
    delta_thetai = args[1]  
    x = args[3]  
    y = args[4]  
    lamta = args[5]  

    labels = set(y)  
    thetai = original_thetai  
    thetai[i, :] = thetai[i, :] - alfa * delta_thetai  
    k = 0  
    sum_log_p = 0.0  
    for label in labels:  
        index = y == label  
        xi = x[index]  
        p = condProb(original_thetai,thetai[k, :], xi)  
        log_p = np.log10(p)  
        sum_log_p = sum_log_p + log_p.sum()  
        k = k + 1  
    # 这就是代价函数的公式
    r = -sum_log_p / x.shape[0]+ (lamta / 2.0) * pow(norm(thetai),2)  
     #print r ,alfa  

    return r  
class Softmax:  
    def __init__(self, alfa, lamda, feature_num, label_mum, run_times = 500, col = 1e-6):  
        self.alfa = alfa  
        self.lamda = lamda  
        self.feature_num = feature_num  
        self.label_num = label_mum  
        self.run_times = run_times  
        self.col = col  
        self.theta = np.random.random((label_mum, feature_num + 1))+1.0  
    def oneDimSearch(self, original_thetai,delta_thetai,i,x,y ,lamta):  
         res = minimize(costFunc, 0.0, method = 'Powell', args =(original_thetai,delta_thetai,i,x,y ,lamta))  
         return res.x  

    def train(self, x, y):
         tmp = np.ones((x.shape[0], x.shape[1] + 1))
         tmp[:,1:tmp.shape[1]] = x
         x = tmp
         del tmp
         labels = set(y)
         self.errors = []
         old_alfa = self.alfa
         for kk in range(0, self.run_times):
             i = 0

             # 因为Softmax中不同标签的样本的权值是不一样的
             # 下面就是更新每个标签样本对应的权值
             for label in labels:
                 tmp_theta = copy.deepcopy(self.theta)
                 one = np.zeros(x.shape[0])
                 index = y == label
                 # one位x.shape[0]行的数组，将分类是对应的类别设置为1
                 one[index] =  1.0
                 thetai = np.array([self.theta[i,:]])
                 #  预测输出结果
                 prob = self.condProb(thetai, x)
                 # 标签与预测结果的差
                 prob = np.array([one - prob])
                 prob = prob.transpose()
                 # 此处为梯度表达式
                 delta_thetai=-np.sum(x*prob, axis=0)/x.shape[0]+self.lamda * self.theta[i, :]
                 #alfa = self.oneDimSearch(self.theta,delta_thetai,i,x,y ,self.lamda)#一维搜索法寻找最优的学习率，没有实现
                 # 更新权值                 
                 self.theta[i, :] = self.theta[i,:] - self.alfa*np.array([delta_thetai])
                 i = i + 1
             self.errors.append(self.performance(tmp_theta))

    def performance(self, tmp_theta):
         return norm(self.theta - tmp_theta)

    def dot(self, a, b):
         mat_dot = np.dot(a, b)
         return np.exp(mat_dot)

    def condProb(self, thetai, xi):
         numerator = self.dot(thetai, xi.transpose())

         denominator = self.dot(self.theta, xi.transpose())
         denominator = np.sum(denominator, axis = 0)
         p = numerator[0] / denominator
         return p

    def predict(self, x):
         tmp = np.ones((x.shape[0], x.shape[1]+1))
         tmp[:,1:tmp.shape[1]] = x
         x = tmp
         row = x.shape[0]
         col = self.theta.shape[0]
         pre_res = np.zeros((row, col))
         for i in range(0, row):
             xi = x[i, :]
             for j in range(0, col):
                 thetai = self.theta[j, :]
                 p = self.condProb(np.array([thetai]), np.array([xi]))
                 pre_res[i, j] = p
         r = []
         for i in range(0, row):
             tmp = []
             line = pre_res[i, :]
             ind = line.argmax()
             tmp.append(ind)
             tmp.append(line[ind])
             r.append(tmp)
         return np.array(r)

    def evaluate(self):
         pass

def sample(sample_num, feature_num, label_num):
    n = int(sample_num / label_num)
    x = np.zeros((n*label_num, feature_num))
    y = np.zeros(n*label_num, dtype = np.int)
    for i in range(0, label_num):
        x[i*n : i*n+n, :] = np.random.random((n,feature_num)) + i
        y[i*n : i*n+n] = i
    return [x,y]

def save(name, x, y):
    writer = open(name, 'w')
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            writer.write(str(x[i,j]) + ' ')
        writer.write(str(y[i])+'\n')
    writer.close()

def load(name):
    x = []
    y = []
    for line in open(name, 'r'):
        ele = line.split(' ')
        tmp = []
        for i in range(0, len(ele) - 1):
            tmp.append(float(ele[i]))
        x.append(tmp)
        y.append(int(ele[len(ele) - 1]))
    return [x,y]

def plotRes(pre, real, test_x, l):
    s = set(pre)
    col = ['r', 'b', 'g', 'y', 'm']
    fig = plt.figure()

    ax = fig.add_subplot(111)
    for i in range(0, len(s)):
        index1 = pre == i
        index2 = real == i
        x1 = test_x[index1, :]
        x2 = test_x[index2, :]
        ax.scatter(x1[:,0], x1[:,1],color=col[i],marker='v',linewidths=0.5)
        ax.scatter(x2[:,0], x2[:,1],color=col[i],marker='.',linewidths=12)
    plt.title('learning rating='+str(1))
    plt.legend('c1:predict','c1:true',
               'c2:predict','c2:true',
               'c3:predict','c3:true',
               'c4:predict','c4:true',
               'c5:predict','c5:true'),shadow = true,loc = (0.01, 0.4)
    plt.show()

if __name__ == '__main__':
    
    """
    数据的准备
    """
    #数据导入与处理
    
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

    df = pandas.read_table("F://project//spectrum-data//20160921_165221_117.7044_38.9908_1.txt",names=["fc","E"])
    cdmaUpD = df[df.fc.between(825,835)]#1
    cdmaUpD["name"] = 1
    cdmaDownD = df[df.fc.between(870,880)]#2
    cdmaDownD["name"] = 2
    egsmUp = df[df.fc.between(885,890)]#3
    egsmUp["name"] = 3
    egsmDown = df[df.fc.between(930,935)]#4
    egsmDown["name"] = 4
    wlan = df[df.fc.between(2400,2483.5)]#5
    wlan["name"] = 5
    wd = df[df.fc.between(3400,3600)]#6
    wd["name"] = 6
    lteY = df[(df.fc.between(1880,1900))|(df.fc.between(2320,2370))|(df.fc.between(2575,2635))]#7
    lteY["name"] = 7
    df1 = pandas.concat([cdmaUpD,cdmaDownD,egsmUp,egsmDown,wlan,lteY])
    
    
    #计算band
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
    cdmaUpD['band'] = band1
    cdmaDownD['band'] = band2
    egsmUp['band'] = band3
    egsmDown['band'] = band4
    wlan['band'] = band5
    lteY['band'] = band6
    z = pandas.concat([cdmaUpD,cdmaDownD,egsmUp,egsmDown,wlan,lteY])
    
    
    
    """
    数据的处理
    """
    [x,y] = [np.array([z['fc'],z['E'],z['band']]).T,np.array(z['name'])]
    #[x,y] = sample(1000,2,5)
    #save('data.txt', x, y)
    #[x,y] = load('data.txt')
    #index= [range(0, len(x))] 
    #random.shuffle(index)
    x = np.array(x)  
    y = np.array(y)   
    #x_train = x[index[0:700],:]
    #y_train = y[index[0:700]]
    #这里讲第二个参数设置为0.0，即不用正则化，因为模型中没有高次项，用正则化反而使效果变差  
    softmax = Softmax(0.4, 0.0, 3, 7)
    softmax.train(x, y)       
    x_test = x[index[7000:9748],:]
    y_test = y[index[7000:9748]]
    r = softmax.predict(x_test)
    plotRes(r[:,0], y_test, x_test, softmax.alfa)
    t = r[:,0] != y_test
    o = np.zeros(len(t))
    o[1] = 1
    err = sum(o)
    
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/')

