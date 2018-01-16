# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:26:00 2017

@author: Administrator
"""
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
class processing(object):


    def __init__(self,filepath):
        
        self.path = filepath
        #self.a = os.listdir(self.path)
        self.cdmaUpD = pandas.DataFrame()
        self.cdmaDownD = pandas.DataFrame()
        self.egsmUp = pandas.DataFrame()
        self.egsmDown = pandas.DataFrame()
        self.wlan = pandas.DataFrame()
        self.lteY = pandas.DataFrame()
        self.df = pandas.read_table(filepath,names=["fc","E"])
    
    
    def seprate(self):
        
        self.cdmaUpD = self.df[self.df.fc.between(825,835)]#1
        self.cdmaUpD["name"] = "1"
        self.cdmaDownD = self.df[self.df.fc.between(870,880)]#2
        self.cdmaDownD["name"] = "2"
        self.egsmUp = self.df[self.df.fc.between(885,890)]#3
        self.egsmUp["name"] = "3"
        self.egsmDown = self.df[self.df.fc.between(930,935)]#4
        self.egsmDown["name"] = "4"
        self.wlan = self.df[self.df.fc.between(2400,2483.5)]#5
        self.wlan["name"] = "5"
        self.wd = self.df[self.df.fc.between(3400,3600)]#6
        self.wd["name"] = "6"
        self.lteY = self.df[(self.df.fc.between(1880,1900))|(self.df.fc.between(2320,2370))|(self.df.fc.between(2575,2635))]#7
        self.lteY["name"] = "7"
        self.df = pandas.concat([self.cdmaUpD,self.cdmaDownD,self.egsmUp,self.egsmDown,self.wlan,self.lteY])
        
    
    
    def band(self,l):
        
        sum1 = 0
        num1 = 0
        l = l.reset_index()
        for i in range(len(l)-1):
            det = l['E'][i]-l['E'][i+1]
            if det>3:
                sum1 += (l['fc'][i]-l['fc'][i+1])
                num1 += 1
        band1 = sum1/num1
        return band1
    
    
    def result(self):
        
        self.cdmaUpD['band'] = self.band(self.cdmaUpD)
        self.cdmaDownD['band'] = self.band(self.cdmaDownD)
        self.egsmUp['band'] = self.band(self.egsmUp)
        self.egsmDown['band'] = self.band(self.egsmDown)
        self.wlan['band'] = self.band(self.wlan)
        self.lteY['band'] = self.band(self.lteY)
        z = pandas.concat([self.cdmaUpD,self.cdmaDownD,self.egsmUp,self.egsmDown,self.wlan,self.lteY])
        return z
    
        
if __name__ == "__main__":
    
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs  
    from mpl_toolkits.mplot3d import Axes3D
    a = os.listdir("F:/project/spectrum-data")
    for i in range(len(a)):
        w = processing("F://project//spectrum-data//"+a[i])
        w.seprate()
        z = w.result()
        D = np.array([z['E'],z['fc'],z['band']])
        D1 = D.transpose() 
        ypred = KMeans(n_clusters=8,random_state=9).fit_predict(D1)
        ax = plt.subplot(111, projection='3d')
        ax.scatter(D1[:, 1], D1[:, 0],D1[:, 2], c=ypred)
        ax.set_zlabel('band') #坐标轴
        ax.set_ylabel('E')
        ax.set_xlabel('fc')
        plt.savefig("F:/project/operatordata/pics/cluster1/%i.png"%i)

        