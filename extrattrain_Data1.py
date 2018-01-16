# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:19:39 2017

@author: Administrator
"""

    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(825,830)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta11=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(3):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        #w = w.T.tolist()
        theta11.append(w[-1])
       
    
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(870,875)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta22=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(3):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta22.append(w[-1])
        
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(885,890)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta33=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(3):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta33.append(w[-1])
       
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(930,935)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta44=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(3):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta44.append(w[-1])
       
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(2400,2405)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta55=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(3):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta55.append(w[-1])
        
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[(df.fc.between(1880,1885))]#6 #1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta66=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(3):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta66.append(w[-1])
        



r11 = theta11[0]
for i in range(1,len(theta11)):
    r11 = np.hstack((r11,theta11[i]))   
 
r22 = theta22[0]
for i in range(1,len(theta22)):
    r22 = np.hstack((r22,theta22[i])) 

r33 = theta33[0]
for i in range(1,len(theta33)):
    r33 = np.hstack((r33,theta33[i])) 
    
r44 = theta44[0]
for i in range(1,len(theta44)):
    r44 = np.hstack((r44,theta44[i])) 
    
r55 = theta55[0]
for i in range(1,len(theta55)):
    r55 = np.hstack((r55,theta55[i])) 
    
r66 = theta66[0]
for i in range(1,len(theta66)):
    r66 = np.hstack((r66,theta66[i])) 
    

rr = np.hstack((r11,r22,r33,r44,r55,r66)).T
R = np.stack((r,rr)).T

