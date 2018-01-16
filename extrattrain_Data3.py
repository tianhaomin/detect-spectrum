# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 20:49:41 2017

@author: Administrator
"""

    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(825,830)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta1111=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(5):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        #w = w.T.tolist()
        theta1111.append(w[-1])
       
    
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(870,875)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta2222=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(5):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta2222.append(w[-1])
        
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(885,890)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta3333=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(5):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta3333.append(w[-1])
       
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(930,935)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta4444=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(5):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta4444.append(w[-1])
       
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(2400,2405)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta5555=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(5):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta5555.append(w[-1])
        
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[(df.fc.between(1880,1885))]#6 #1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta6666=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(5):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta6666.append(w[-1])
        



r1111 = theta1111[0]
for i in range(1,len(theta1111)):
    r1111 = np.hstack((r1111,theta1111[i]))   
 
r2222 = theta2222[0]
for i in range(1,len(theta2222)):
    r2222 = np.hstack((r2222,theta2222[i])) 

r3333 = theta3333[0]
for i in range(1,len(theta3333)):
    r3333 = np.hstack((r3333,theta3333[i])) 
    
r4444 = theta4444[0]
for i in range(1,len(theta4444)):
    r4444 = np.hstack((r4444,theta4444[i])) 
    
r5555 = theta5555[0]
for i in range(1,len(theta5555)):
    r5555 = np.hstack((r5555,theta5555[i])) 
    
r6666 = theta6666[0]
for i in range(1,len(theta6666)):
    r6666 = np.hstack((r6666,theta6666[i])) 
    

rrrr = np.hstack((r1111,r2222,r3333,r4444,r5555,r6666)).T
R1 = np.stack((r,rr,rrr,rrrr)).T

