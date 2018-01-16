# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 08:45:47 2017

@author: Administrator
"""

import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn
import os

def projection(A,b):
    ####
    # return A*inv(AT*A)*AT*b
    ####
    AA = A.T.dot(A)
    w=np.linalg.inv(AA).dot(A.T).dot(b)
    print (w)
    return A.dot(w)

def projection1(A,b):
    ####
    # return A*inv(AT*A)*AT*b
    ####
    AA = A.T.dot(A)
    w=np.linalg.inv(AA).dot(A.T).dot(b)
    return w

a = os.listdir("F:/project/Yin/spectrum-data")
df = pandas.DataFrame()


    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//Yin//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(825,830)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta1=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(2):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        #w = w.T.tolist()
        theta1.append(w[1])
        yw = projection(A,b)
        yw.shape = (yw.shape[0],)
       '''
        plt.figure(figsize=(8,4))
        plt.plot(X,Y,color='m',linestyle='',marker='o',label=u"points")
        plt.plot(X,yw,color='r',linestyle='-',marker='.',label=u"fitted")
        #plt.savefig('F:/project/operatordata/pic1/fitCdmaUp/%i.png'%i)
        plt.legend()
        plt.show()
        plt.figure()
       '''



    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(870,875)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta2=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(2):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta2.append(w[1])
        yw = projection(A,b)
        yw.shape = (yw.shape[0],)
      '''  
        plt.figure(figsize=(8,4))
        plt.plot(X,Y,color='m',linestyle='',marker='o',label=u"points")
        plt.plot(X,yw,color='r',linestyle='-',marker='.',label=u"fitted")
        #plt.savefig('F:/project/operatordata/pic1/fitCdmaUp/%i.png'%i)
        plt.legend()
        plt.show()
        plt.figure()
       
   '''


##########

    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(885,890)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta3=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(2):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta3.append(w[1])
        yw = projection(A,b)
        yw.shape = (yw.shape[0],)
        
       ''' plt.figure(figsize=(8,4))
        plt.plot(X,Y,color='m',linestyle='',marker='o',label=u"points")
        plt.plot(X,yw,color='r',linestyle='-',marker='.',label=u"fitted")
        #plt.savefig('F:/project/operatordata/pic1/fitCdmaUp/%i.png'%i)
        plt.legend()
        plt.show()
        plt.figure()
       '''





    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(930,935)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta4=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(2):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta4.append(w[1])
        yw = projection(A,b)
        yw.shape = (yw.shape[0],)
        '''
        plt.figure(figsize=(8,4))
        plt.plot(X,Y,color='m',linestyle='',marker='o',label=u"points")
        plt.plot(X,yw,color='r',linestyle='-',marker='.',label=u"fitted")
        #plt.savefig('F:/project/operatordata/pic1/fitCdmaUp/%i.png'%i)
        plt.legend()
        plt.show()
        plt.figure()
        '''





    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(2400,2405)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta5=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(2):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta5.append(w[1])
        yw = projection(A,b)
        yw.shape = (yw.shape[0],)
        '''
        plt.figure(figsize=(8,4))
        plt.plot(X,Y,color='m',linestyle='',marker='o',label=u"points")
        plt.plot(X,yw,color='r',linestyle='-',marker='.',label=u"fitted")
        #plt.savefig('F:/project/operatordata/pic1/fitCdmaUp/%i.png'%i)
        plt.legend()
        plt.show()
        plt.figure()
        
'''




    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[(df.fc.between(1880,1885))]#6 #1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta6=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(2):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta6.append(w[1])
        yw = projection(A,b)
        yw.shape = (yw.shape[0],)
        '''
        plt.figure(figsize=(8,4))
        plt.plot(X,Y,color='m',linestyle='',marker='o',label=u"points")
        plt.plot(X,yw,color='r',linestyle='-',marker='.',label=u"fitted")
        #plt.savefig('F:/project/operatordata/pic1/fitCdmaUp/%i.png'%i)
        plt.legend()
        plt.show()
        plt.figure()
        '''

label1 = np.zeros(850).astype(int)
label2 = np.ones(850).astype(int)
label3 = np.ones(850).astype(int)*2
label4 = np.ones(850).astype(int)*3
label5 = np.ones(850).astype(int)*4
label6 = np.ones(850).astype(int)*5
label = np.hstack((label1,label2,label3,label4,label5,label6))



'''
label = np.array([1,0,0,0,0,0]).T
for i in range(29):
    a = np.array([1,0,0,0,0,0]).T
    label = np.hstack((label,a)).T
label.shape=(180,6)

for i in range(30):
    a = np.array([0,1,0,0,0,0])
    label = np.hstack((label,a))
    
for i in range(30):
    a = np.array([0,0,1,0,0,0])
    label = np.hstack((label,a))
    
for i in range(30):
    a = np.array([0,0,0,1,0,0])
    label = np.hstack((label,a))
    
for i in range(30):
    a = np.array([0,0,0,0,1,0])
    label = np.hstack((label,a))
    
for i in range(30):
    a = np.array([0,0,0,0,0,1])
    label = np.hstack((label,a))
'''
    
    
r1 = theta1[0]
for i in range(1,len(theta1)):
    r1 = np.hstack((r1,theta1[i]))   
 
r2 = theta2[0]
for i in range(1,len(theta2)):
    r2 = np.hstack((r2,theta2[i])) 

r3 = theta3[0]
for i in range(1,len(theta3)):
    r3 = np.hstack((r3,theta3[i])) 
    
r4 = theta4[0]
for i in range(1,len(theta4)):
    r4 = np.hstack((r4,theta4[i])) 
    
r5 = theta5[0]
for i in range(1,len(theta5)):
    r5 = np.hstack((r5,theta5[i])) 
    
r6 = theta6[0]
for i in range(1,len(theta6)):
    r6 = np.hstack((r6,theta6[i])) 
    

r = np.hstack((r1,r2,r3,r4,r5,r6)).T
w
r = r.T   
label = np.stack((label1,label2,label3,label4,label5,label6))    
label.shape=(180,1)
    
r[0]    
    
    
#nnet样本数量与预测的关系
plt.figure(figsize=(8,4))
plt.plot([30,100,200,400,700],[0.3389,0.3483,0.3308,0.40,0.4238],color='r',linestyle='-',marker='.',label=u"training error")
plt.plot([30,100,200,400,700],[0.34,0.3867,0.3517,0.3967,0.4433],color='g',linestyle='-',marker='.',label=u"predict error")
plt.xlabel('num of training exp')
plt.ylabel('training error')
plt.legend()
plt.show()
plt.figure()
#nnet拟合次数与预测准确度的关系
plt.figure(figsize=(8,4))
plt.plot([1,2,5,10],[0.3308,0.6708,0.833,0.7133],color='r',linestyle='-',marker='.',label=u"training error")
plt.plot([1,2,5,10],[0.3517,0.6700,0.833,0.7262],color='g',linestyle='-',marker='.',label=u"predict error")
plt.xlabel('fitting order')
plt.ylabel('error')
plt.legend()
plt.show()
plt.figure()
#keras分类（一次）d迭代次数（样本数850）与准确度的关系
plt.figure(figsize=(8,4))
plt.plot([100,1000,3000,5000,10000,15000],[0.47269,0.56,0.57,0.59008,0.57,0.498],color='g',linestyle='-',marker='.')
plt.xlabel('num of iter')
plt.ylabel('acc')
plt.title("keras classfication")
plt.legend()
plt.show()
plt.figure()
#keras分类样本100拟合次数与准确率的关系
plt.figure(figsize=(8,4))
plt.plot([1,2,4,9],[0.7269,0.3238,0.1517,0.1667],color='r',linestyle='-',marker='.')
plt.xlabel('fitting order')
plt.ylabel('acc')
plt.title("order of the fitting & accurace")
plt.legend()
plt.show()
plt.figure()
#keras分类均值分布（10）acc与迭代次数
plt.figure(figsize=(8,4))
plt.plot([100,1000,3000,5000,15000],[0.3003,0.7003,0.7535,0.7950,0.8863],color='g',linestyle='-',marker='.',label=u"ten")
plt.plot([100,1000,3000,5000,15000],[0.3003,0.6619,0.71,0.7500,0.7207],color='r',linestyle='-',marker='.',label=u"five")
plt.plot([100,1000,3000,5000,15000],[0.72,0.7658,0.91,0.9443,0.9826],color='y',linestyle='-',marker='.',label=u"twenty")
plt.xlabel('num of iter')
plt.ylabel('acc')
plt.title("keras means distribution classfication")
plt.legend()
plt.show()
plt.figure() 
#keras 均值分类（时间一定，随机过程）  
plt.figure(figsize=(8,4))
plt.plot([100,1000,3000,5000,15000],[0.3,0.4,0.46,0.57,0.64],color='r',linestyle='-',marker='.',label=u"mean")
plt.xlabel("num of iter")
plt.ylabel("acc")
plt.title("keras mean classfication")
plt.legend()
plt.show()
#keras方差分类
plt.figure(figsize=(8,4))
plt.plot([100,1000,3000,5000,15000],[0.3,0.4,0.56,0.57,0.60],color='g',linestyle='-',marker='.',label=u"var")
plt.xlabel("num of iter")
plt.ylabel("acc")
plt.title("keras var classification")
plt.legend()
plt.show()
#keras分类方差分布
plt.figure(figsize=(8,4))
plt.plot([100,1000,3000,5000,15000],[0.4,0.68,0.91,0.93,0.986],color='r',linestyle='-',marker='.',label=u'k=20')
plt.xlabel("num of iter")
plt.ylabel("acc")
plt.title("keras var distribution classification")
plt.legend()
plt.show()


plt.figure(figsize=(12,12))
plt.plot([1000,2000,3000,4000,5000,15000],[0.578,0.587,0.589,0.5895,0.589,0.59],color='g',linestyle='-',marker='.',label=u"1")
plt.plot([1000,2000,3000,4000,5000,15000],[0.664,0.669,0.668,0.6900,0.690,0.70],color='r',linestyle='-',marker='.',label=u"2")
plt.plot([1000,2000,3000,4000,5000,15000],[0.751,0.870,0.880,0.8940,0.900,0.91],color='y',linestyle='-',marker='.',label=u"3")
plt.plot([1000,2000,3000,4000,5000,15000],[0.778,0.873,0.889,0.8990,0.9025,0.9182],color='b',linestyle='-',marker='.',label=u"4")
plt.plot([1000,2000,3000,4000,5000,15000],[0.785,0.876,0.881,0.8944,0.9026,0.9162],color='#A52A2A',linestyle='-',marker='.',label=u"5")
plt.plot([1000,2000,3000,4000,5000,15000],[0.750,0.871,0.882,0.8945,0.8961,0.9129],color='#D3D3D3',linestyle='-',marker='.',label=u"5+9")
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title("Fit times distribution classfication")
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot([200,1000,2000,3000,5000,15000],[0.58,0.71,0.8,0.84,0.88,0.93],color='g',linestyle='-',marker='.',label=u"K=10")
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title("Epoches distribution classfication")
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
plt.plot([200,1000,2000,3000,5000,10000,15000],[0.518,0.61,0.75,0.796,0.818,0.824,0.853],color='g',linestyle='-',marker='.',label=u"K=23")
plt.plot([200,1000,2000,3000,5000,10000,15000],[0.647,0.769,0.845,0.89,0.935,0.95,0.96],color='r',linestyle='-',marker='.',label=u"K=20")
plt.plot([200,1000,2000,3000,5000,10000,15000],[0.63,0.728,0.794,0.84,0.89,0.93,0.948],color='y',linestyle='-',marker='.',label=u"K=17")
plt.plot([200,1000,2000,3000,5000,10000,15000],[0.43,0.53,0.58,0.61,0.657,0.693,0.71],color='b',linestyle='-',marker='.',label=u"K=15")
plt.plot([200,1000,2000,3000,5000,10000,15000],[0.38,0.471,0.51,0.54,0.58,0.61,0.63],color='#A52A2A',linestyle='-',marker='.',label=u"K=13")
plt.plot([200,1000,2000,3000,5000,10000,15000],[0.35,0.42,0.45,0.484,0.515,0.53,0.55],color='#D3D3D3',linestyle='-',marker='.',label=u"K=10")
plt.xlabel('epoch')
plt.ylabel('acc')
#plt.title("Epoches distribution classfication")
plt.legend()
plt.show()





#话confusion matrix
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

test_Y_hat = model.predict(X_test, batch_size=256)
conf = np.zeros([6,6])
confnorm = np.zeros([6,6])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,6):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
#相关性分析
plot_confusion_matrix(confnorm, labels=["cdmaUp","cdmaDown","egsmUp","egsmDown","wlan","lte"])
plt.figure(figsize=(8,4))
plt.plot([100,1000,3000,5000,15000],[0.4,0.68,0.77,0.773,0.7786],color='r',linestyle='-',marker='.')
plt.xlabel("num of iter")
plt.ylabel("acc")
#plt.title("keras var distribution classification")
plt.legend()
plt.show()