# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:31:16 2017

@author: Administrator
"""
from numpy import *
import matplotlib.pyplot as plt
from numpy import *
# 对某一点计算估计值
def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for i in range(m):
        diffMat = testPoint - xMat[i, :]
        weights[i, i] = exp(diffMat * diffMat.T/(-2.0 * k**2))      # 计算权重对角矩阵
    xTx = xMat.T * (weights * xMat)                                 # 奇异矩阵不能计算
    if linalg.det(xTx) == 0.0:
        print('This Matrix is singular, cannot do inverse')
        return
    theta = xTx.I * (xMat.T * (weights * yMat))                     # 计算回归系数
    return testPoint * theta

# 对所有点计算估计值
def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

x = z[0]["fc"]  
xArr = np.array(x)  
y1 = z[0]['E']
yArr = np.array(y1)


yHat = lwlrTest(xArr, xArr, yArr, 0.01)
xMat = mat(xArr)
strInd = xMat[:, 1].argsort(0)
xSort = xMat[strInd][:, 0, :]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(xSort[:, 1], yHat[strInd])
ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s = 2, c = 'red')
plt.show()