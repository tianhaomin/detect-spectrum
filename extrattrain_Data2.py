# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:10:00 2017

@author: Administrator
"""

    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(825,830)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta111=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(4):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        #w = w.T.tolist()
        theta111.append(w[-1])
       
    
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(870,875)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta222=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(4):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta222.append(w[-1])
        
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(885,890)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta333=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(4):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta333.append(w[-1])
       
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(930,935)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta444=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(4):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta444.append(w[-1])
       
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(2400,2405)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta555=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(4):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta555.append(w[-1])
        
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[(df.fc.between(1880,1885))]#6 #1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta666=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(4):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta666.append(w[-1])
        



r111 = theta111[0]
for i in range(1,len(theta111)):
    r111 = np.hstack((r111,theta111[i]))   
 
r222 = theta222[0]
for i in range(1,len(theta222)):
    r222 = np.hstack((r222,theta222[i])) 

r333 = theta333[0]
for i in range(1,len(theta333)):
    r333 = np.hstack((r333,theta333[i])) 
    
r444 = theta444[0]
for i in range(1,len(theta444)):
    r444 = np.hstack((r444,theta444[i])) 
    
r555 = theta555[0]
for i in range(1,len(theta555)):
    r555 = np.hstack((r555,theta555[i])) 
    
r666 = theta666[0]
for i in range(1,len(theta666)):
    r666 = np.hstack((r666,theta666[i])) 
    

rrr = np.hstack((r111,r222,r333,r444,r555,r666)).T
R = np.stack((r,rr,rrr)).T

