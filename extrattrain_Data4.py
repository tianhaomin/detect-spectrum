# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 08:37:22 2017

@author: Administrator
"""

    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(825,830)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta11111=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(6):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        #w = w.T.tolist()
        theta11111.append(w[-1])
       
    
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(870,875)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta22222=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(6):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta22222.append(w[-1])
        
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(885,890)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta33333=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(6):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta33333.append(w[-1])
       
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(930,935)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta44444=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(6):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta44444.append(w[-1])
       
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[df.fc.between(2400,2405)]#1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta55555=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(6):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta55555.append(w[-1])
        
        
        
    z1=[]
    for i in range(850):
        df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
        df1 = df[(df.fc.between(1880,1885))]#6 #1
        df1['fc'] = df1['fc']
        z1.append(df1)
    theta66666=[]
    for i in range(len(z1)) :
        X = np.array(z1[i]['fc'])
        Y = np.array(z1[i]['E'])
        b = Y.reshape(Y.shape[0],1)
        m = []
        for i in range(6):
            m.append(X**(i))  
        A = np.array(m).T
        w = projection1(A,b)
        theta66666.append(w[-1])
        



r11111 = theta11111[0]
for i in range(1,len(theta11111)):
    r11111 = np.hstack((r11111,theta11111[i]))   
 
r22222 = theta22222[0]
for i in range(1,len(theta22222)):
    r22222 = np.hstack((r22222,theta22222[i])) 

r33333 = theta33333[0]
for i in range(1,len(theta33333)):
    r33333 = np.hstack((r33333,theta33333[i])) 
    
r44444 = theta44444[0]
for i in range(1,len(theta44444)):
    r44444 = np.hstack((r44444,theta44444[i])) 
    
r55555 = theta55555[0]
for i in range(1,len(theta55555)):
    r55555 = np.hstack((r55555,theta55555[i])) 
    
r66666 = theta66666[0]
for i in range(1,len(theta66666)):
    r66666 = np.hstack((r66666,theta66666[i])) 
    
rrrrr = np.hstack((r11111,r22222,r33333,r44444,r55555,r66666)).T




R2 = np.stack((r,rr,rrr,rrrr,rrrrr)).T