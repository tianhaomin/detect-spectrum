# -*- coding: utf-8 -*-
"""
Created on Sat May  6 16:27:51 2017

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

a = os.listdir("F:/project/spectrum-data")
df = pandas.DataFrame()


    z1=[]
    for i in range(700,800):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(825,835)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta1=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(11):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        #w = w.T.tolist()
        theta1.append(w)
        yw = projection(A,b)
        yw.shape = (yw.shape[0],)
       
        



    z1=[]
    for i in range(700,800):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(870,880)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta2=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(11):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta2.append(w)
        yw = projection(A,b)
        yw.shape = (yw.shape[0],)
        
   




    z1=[]
    for i in range(700,800):
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
        for i in range(11):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta3.append(w)
        yw = projection(A,b)
        yw.shape = (yw.shape[0],)
        
       
       





    z1=[]
    for i in range(700,800):
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
        for i in range(11):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta4.append(w)
        yw = projection(A,b)
        yw.shape = (yw.shape[0],)
       
        





    z1=[]
    for i in range(700,800):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(2400,2483.5)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta5=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(11):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta5.append(w)
        yw = projection(A,b)
        yw.shape = (yw.shape[0],)
        
        





    z1=[]
    for i in range(700,800):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[(df.fc.between(1880,1900))|(df.fc.between(2320,2370))|(df.fc.between(2575,2635))]#6 #1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta6=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(11):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta6.append(w)
        yw = projection(A,b)
        yw.shape = (yw.shape[0],)
        
       
        

l1 = np.zeros(100).astype(int)
l2 = np.ones(100).astype(int)
l3 = np.ones(100).astype(int)*2
l4 = np.ones(100).astype(int)*3
l5 = np.ones(100).astype(int)*4
l6 = np.ones(100).astype(int)*5
l = np.hstack((l1,l2,l3,l4,l5,l6))



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
    
    
k1 = theta1[0]
for i in range(1,len(theta1)):
    k1 = np.hstack((k1,theta1[i]))   
 
k2 = theta2[0]
for i in range(1,len(theta2)):
    k2 = np.hstack((k2,theta2[i])) 

k3 = theta3[0]
for i in range(1,len(theta3)):
    k3 = np.hstack((k3,theta3[i])) 
    
k4 = theta4[0]
for i in range(1,len(theta4)):
    k4 = np.hstack((k4,theta4[i])) 
    
k5 = theta5[0]
for i in range(1,len(theta5)):
    k5 = np.hstack((k5,theta5[i])) 
    
k6 = theta6[0]
for i in range(1,len(theta6)):
    k6 = np.hstack((k6,theta6[i])) 
    
 
k = np.hstack((k1,k2,k3,k4,k5,k6)).T