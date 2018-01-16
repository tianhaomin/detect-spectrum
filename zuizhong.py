# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:10:36 2017

@author: Administrator
"""

import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
#拟合函数
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
#数据提取
a = os.listdir("F:/project/Yin/spectrum-data")
df = pandas.DataFrame()
z1=[]
for i in range(10):
        df = pandas.read_table("F:/project/Yin/spectrum-data//"+a[i],names=["fc","E"])
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
z1=[]
for i in range(10):
        df = pandas.read_table("F:/project/Yin/spectrum-data//"+a[i],names=["fc","E"])
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
z1=[]
for i in range(10):
        df = pandas.read_table("F:/project/Yin/spectrum-data//"+a[i],names=["fc","E"])
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
z1=[]
for i in range(10):
        df = pandas.read_table("F:/project/Yin/spectrum-data//"+a[i],names=["fc","E"])
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
z1=[]
for i in range(10):
        df = pandas.read_table("F:/project/Yin/spectrum-data//"+a[i],names=["fc","E"])
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
z1=[]
for i in range(10):
        df = pandas.read_table("F:/project/Yin/spectrum-data//"+a[i],names=["fc","E"])
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

label1 = np.zeros(10).astype(int)
label2 = np.ones(10).astype(int)
label3 = np.ones(10).astype(int)*2
label4 = np.ones(10).astype(int)*3
label5 = np.ones(10).astype(int)*4
label6 = np.ones(10).astype(int)*5
label = np.hstack((label1,label2,label3,label4,label5,label6))
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
#label.shape=(180,1)
#分类模型构建
X = r
Y = label

# encode class values as integers
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
# convert integers to dummy variables (one hot encoding)
dummy_y = np_utils.to_categorical(encoded_Y)
# splitting data into training set and test set. If random_state is set to an integer, the split datasets are fixed.
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3, random_state=0)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.add(Dense(1200, activation='relu', input_dim=1))
model.add(Dropout(0.2))
#model.add(Dense(1200, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          epochs=15,
          batch_size=128)


#测试测试
score = model.evaluate(X_test, Y_test, batch_size=256)
q=model.predict(X_test,batch_size=128)#预测


#模糊矩阵
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
plot_confusion_matrix(confnorm, labels=["cdmaUp","cdmaDown","egsmUp","egsmDown","wlan","lte"])