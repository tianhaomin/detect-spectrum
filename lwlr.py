# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:37:44 2017

@author: Administrator
"""
#数据读入
import numpy as np
from numpy import *
import pandas
import matplotlib.pyplot as plt
import seaborn


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    print (numFeat)
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


#局部加权回归
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    #print(m)
    weights = mat(eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws



def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy



path1 = ['F://project/groupData/cdmaUpD/1.csv','F://project/groupData/cdmaDownD/1.csv',
        'F://project/groupData/egsmUp/1.csv','F://project/groupData/egsmDown/1.csv',
        'F://project/groupData/wlan/1.csv','F://project/groupData/lteY/1.csv']

path2 = ['F://project/groupData/cdmaDownD/0.csv','F://project/groupData/cdmaDownD/5.csv',
         'F://project/groupData/cdmaDownD/20.csv','F://project/groupData/cdmaDownD/30.csv',
         'F://project/groupData/cdmaDownD/50.csv','F://project/groupData/cdmaDownD/100.csv',
         'F://project/groupData/cdmaDownD/200.csv','F://project/groupData/cdmaDownD/150.csv',
         'F://project/groupData/cdmaDownD/300.csv','F://project/groupData/cdmaDownD/400.csv',
         'F://project/groupData/cdmaDownD/650.csv','F://project/groupData/cdmaDownD/500.csv',
         'F://project/groupData/cdmaDownD/700.csv','F://project/groupData/cdmaDownD/800.csv']

path3 = ['F://project/groupData/egsmUp/0.csv','F://project/groupData/egsmUp/5.csv',
         'F://project/groupData/egsmUp/20.csv','F://project/groupData/egsmUp/30.csv',
         'F://project/groupData/egsmUp/50.csv','F://project/groupData/egsmUp/100.csv',
         'F://project/groupData/egsmUp/200.csv','F://project/groupData/egsmUp/150.csv',
         'F://project/groupData/egsmUp/300.csv','F://project/groupData/egsmUp/400.csv',
         'F://project/groupData/egsmUp/650.csv','F://project/groupData/egsmUp/500.csv',
         'F://project/groupData/egsmUp/700.csv','F://project/groupData/egsmUp/800.csv']

path4 = ['F://project/groupData/egsmDown/0.csv','F://project/groupData/egsmDown/5.csv',
         'F://project/groupData/egsmDown/20.csv','F://project/groupData/egsmDown/30.csv',
         'F://project/groupData/egsmDown/50.csv','F://project/groupData/egsmDown/100.csv',
         'F://project/groupData/egsmDown/200.csv','F://project/groupData/egsmDown/150.csv',
         'F://project/groupData/egsmDown/300.csv','F://project/groupData/egsmDown/400.csv',
         'F://project/groupData/egsmDown/650.csv','F://project/groupData/egsmDown/500.csv',
         'F://project/groupData/egsmDown/700.csv','F://project/groupData/egsmDown/800.csv']

path5 = ['F://project/groupData/wlan/0.csv','F://project/groupData/wlan/5.csv',
         'F://project/groupData/wlan/20.csv','F://project/groupData/wlan/30.csv',
         'F://project/groupData/wlan/50.csv','F://project/groupData/wlan/100.csv',
         'F://project/groupData/wlan/200.csv','F://project/groupData/wlan/150.csv',
         'F://project/groupData/wlan/300.csv','F://project/groupData/wlan/400.csv',
         'F://project/groupData/wlan/650.csv','F://project/groupData/wlan/500.csv',
         'F://project/groupData/wlan/700.csv','F://project/groupData/wlan/800.csv']

path6 = ['F://project/groupData/ltey/0.csv','F://project/groupData/ltey/5.csv',
         'F://project/groupData/ltey/20.csv','F://project/groupData/ltey/30.csv',
         'F://project/groupData/ltey/50.csv','F://project/groupData/ltey/100.csv',
         'F://project/groupData/ltey/200.csv','F://project/groupData/ltey/150.csv',
         'F://project/groupData/ltey/300.csv','F://project/groupData/ltey/400.csv',
         'F://project/groupData/ltey/650.csv','F://project/groupData/ltey/500.csv',
         'F://project/groupData/ltey/700.csv','F://project/groupData/ltey/800.csv']

for i in path6:
    df = pandas.read_csv(i);
    xArr = []
    yArr = []
    r = df.values;
    for i in r[:,0]:
        xArr.append([i])
    for j in r[:,1]:
        yArr.append(j)
    yHat = lwlrTest(xArr, xArr, yArr, 0.07);
    xMat = mat(xArr);
    strInd = xMat[:, 0].argsort(0);
    xSort = xMat[strInd][:, 0, :];
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xSort[:, 0], yHat[strInd])
    ax.scatter(xMat[:, 0].flatten().A[0], mat(yArr).T.flatten().A[0], s = 2, c = 'red')
    #plt.savefig("F:/project/operatordata/pic3/lwlrCdmaUp/%i.png"%i)
    plt.show()
