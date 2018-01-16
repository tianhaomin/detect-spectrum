# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:05:43 2017

@author: Administrator
"""

    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//Yin//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(825,830)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta111111=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(10):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        #w = w.T.tolist()
        theta111111.append(w[-1])
       
    
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(870,875)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta222222=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(10):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta222222.append(w[-1])
        
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(885,890)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta333333=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(10):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta333333.append(w[-1])
       
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(930,935)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta444444=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(10):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta444444.append(w[-1])
       
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(2400,2405)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta555555=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(10):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta555555.append(w[-1])
        
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[(df.fc.between(1880,1885))]#6 #1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta666666=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(10):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta666666.append(w[-1])
        



r111111 = theta111111[0]
for i in range(1,len(theta111111)):
    r111111 = np.hstack((r111111,theta111111[i]))   
 
r222222 = theta222222[0]
for i in range(1,len(theta222222)):
    r222222 = np.hstack((r222222,theta222222[i])) 

r333333 = theta333333[0]
for i in range(1,len(theta333333)):
    r333333 = np.hstack((r333333,theta333333[i])) 
    
r444444 = theta444444[0]
for i in range(1,len(theta444444)):
    r444444 = np.hstack((r444444,theta444444[i])) 
    
r555555 = theta555555[0]
for i in range(1,len(theta555555)):
    r555555 = np.hstack((r555555,theta555555[i])) 
    
r666666 = theta666666[0]
for i in range(1,len(theta666666)):
    r666666 = np.hstack((r666666,theta666666[i])) 
    
rrrrrr = np.hstack((r111111,r222222,r333333,r444444,r555555,r666666)).T




R3 = np.stack((r,rr,rrr,rrrr,rrrrr,rrrrrr)).T
