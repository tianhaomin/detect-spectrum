# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:21:05 2017

@author: Administrator
"""

import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
a = os.listdir("F:/project/spectrum-data")
cdmaUpD = pandas.DataFrame()
cdmaDownD = pandas.DataFrame()
egsmUp = pandas.DataFrame()
egsmDown = pandas.DataFrame()
wlan = pandas.DataFrame()
lteY = pandas.DataFrame()
df = pandas.DataFrame()
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    cdmaUpD = df[df.fc.between(825,830)]#2    plt.figure(figsize=(8,4))
    plt.plot(cdmaDownD['fc'],cdmaDownD['E'])
    plt.xlabel('fc')
    plt.ylabel('E')
    plt.savefig("F:/project/operatordata/pic2/samplecdmaUpD/%i.png"%i)
    plt.show()
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    cdmaDownD = df[df.fc.between(870,875)]#2    plt.figure(figsize=(8,4))
    plt.plot(cdmaDownD['fc'],cdmaDownD['E'])
    plt.xlabel('fc')
    plt.ylabel('E')
    plt.savefig("F:/project/operatordata/pic2/samplecdmaDownD/%i.png"%i)
    plt.show()
       

for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    egsmUp = df[df.fc.between(885,890)]#3
    plt.figure(figsize=(8,4))
    plt.plot(egsmUp['fc'],egsmUp['E'])
    plt.xlabel('fc')
    plt.ylabel('E')
    plt.savefig("F:/project/operatordata/pic2/sampleegsmUp/%i.png"%i)
    plt.show()
    
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    egsmDown = df[df.fc.between(930,935)]#4 
    plt.figure(figsize=(8,4))
    plt.plot(egsmDown['fc'],egsmDown['E'])
    plt.xlabel('fc')
    plt.ylabel('E')
    plt.savefig("F:/project/operatordata/pic2/sampleegsmDown/%i.png"%i)
    plt.show()
    
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    wlan = df[df.fc.between(2400,2405)]#5 
    plt.figure(figsize=(8,4))
    plt.plot(wlan['fc'],wlan['E'])
    plt.xlabel('fc')
    plt.ylabel('E')
    plt.savefig("F:/project/operatordata/pic2/samplewlan/%i.png"%i)
    plt.show()
    
for i in range(len(a)):
    df = pandas.read_table("F://project//spectrum-data//"+a[i],names=["fc","E"])
    lteY = df[df.fc.between(1880,1885)]#6 
    #plt.figure(figsize=(8,4))
    plt.plot(lteY['fc'],lteY['E'])
    plt.xlabel('fc')
    plt.ylabel('E')
    plt.savefig("F:/project/operatordata/pic2/sampleltey/%i.png"%i)
    plt.show()
   
    
'''
def seprate(d):
    cdmaUpD = d[d.fc.between(825,835)]#1
    cdmaDownD = d[d.fc.between(870,880)]#2
    egsmUp = d[d.fc.between(885,890)]#3
    egsmDown = d[d.fc.between(930,935)]#4
    wlan = d[d.fc.between(2400,2483.5)]#5
    lteY = d[(d.fc.between(1880,1900))|(d.fc.between(2320,2370))|(d.fc.between(2575,2635))]#6
    cdmaUpD = pandas.DataFrame()
    cdmaDownD = pandas.DataFrame()
    egsmUp = pandas.DataFrame()
    egsmDown = pandas.DataFrame()
    wlan = pandas.DataFrame()
    lteY = pandas.DataFrame()
    
def draw(d):
    for i in range(len(d)):
        plt.figure(figsize=(8,4))
        plt.plot(d['fc'],d['E'])
        plt.xlabel('fc')
        plt.ylabel('E')
        plt.savefig("F:/project/operatordata/pics/sample%i/%i.png"%i)
        plt.label("%i")
        plt.show()

 '''       


    