# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 16:52:28 2017

@author: Administrator
"""
import seaborn
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
a = os.listdir("F:/project/spectrum-data")
R1 = []
R2 = []
R3 = []
R4 = []
R5 = []
R6 = []
for i in range(800):
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
    for i in range(len(cdmaUpD1)):
        for j in range(len(cdmaUpD2)):
            sum1 = cdmaUpD1['E'][i] + cdmaUpD2['E'][j]
    r1 = sum1/(len(cdmaUpD1)*len(cdmaUpD2))
    R1.append(r1)
    
    cdmaDownD1 = cdmaDownD1.reset_index()
    cdmaDownD2 = cdmaDownD2.reset_index()
    for i in range(len(cdmaDownD1)):
        for j in range(len(cdmaDownD2)):
            sum2 = cdmaDownD1['E'][i] + cdmaDownD2['E'][j]
    r2 = sum2/(len(cdmaDownD1)*len(cdmaDownD2))
    R2.append(r2)
    
    egsmUp1 = egsmUp1.reset_index()
    egsmUp2 = egsmUp2.reset_index()
    for i in range(len(egsmUp1)):
        for j in range(len(egsmUp2)):
            sum3 = egsmUp1['E'][i] + egsmUp2['E'][j]
    r3 = sum3/(len(egsmUp1)*len(egsmUp2))
    R3.append(r3)
    
    egsmDown1 = egsmDown1.reset_index()
    egsmDown2 = egsmDown2.reset_index()
    for i in range(len(egsmDown1)):
        for j in range(len(egsmDown2)):
            sum4 = egsmDown1['E'][i] + egsmDown2['E'][j]
    r4 = sum4/(len(egsmDown1)*len(egsmDown2))
    R4.append(r4)
    
    wlan1 = wlan1.reset_index()
    wlan2 = wlan2.reset_index()
    for i in range(len(wlan1)):
        for j in range(len(wlan2)):
            sum5 = wlan1['E'][i] + wlan2['E'][j]
    r5 = sum5/(len(wlan1)*len(wlan2))
    R5.append(r5)
    
    lteY1 = lteY1.reset_index()
    lteY2 = lteY2.reset_index()
    for i in range(len(lteY1)):
        for j in range(len(lteY2)):
            sum6 = lteY1['E'][i] + lteY2['E'][j]
    r6 = sum6/(len(lteY1)*len(lteY2))
    R6.append(r6)
    
R1 = np.array(R1)/R1.max()
R2 = np.array(R2)/R2.max()
R3 = np.array(R3)/R3.max()
R4 = np.array(R4)/R4.max()
R5 = np.array(R5)/R5.max()
R6 = np.array(R6)/R6.max()
R = np.hstack((R1,R2,R3,R4,R5,R6))
label1 = np.zeros(800).astype(int)
label2 = np.ones(800).astype(int)
label3 = np.ones(800).astype(int)*2
label4 = np.ones(800).astype(int)*3
label5 = np.ones(800).astype(int)*4
label6 = np.ones(800).astype(int)*5
label = np.hstack((label1,label2,label3,label4,label5,label6))


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder    

X = Z
Y = label

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
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.add(Dense(1200, activation='relu', input_dim=2))
model.add(Dropout(0.2))
#model.add(Dense(1200, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              #optimizer=sgd,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          epochs=15000,
          batch_size=256)



score = model.evaluate(X_test, Y_test, batch_size=256)
q=model.predict(X_test,batch_size=256)#预测


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
