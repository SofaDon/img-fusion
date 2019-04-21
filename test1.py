import math
import numpy as np  
from numpy import * 
import matplotlib.pyplot as plt

def cg(A,b,x): #共轭斜量法
    r = b-np.dot(A,x)      #r=b-Ax         r也是是梯度方向
    p= np.copy(r)
    i=0
    while(max(abs(r))>1.e-10 and i < 100  ): 
        #print('i',i)
        #print('r',r)
        pap=np.dot(np.dot(A,p),p)
        if pap==0:                  #分母太小时跳出循环
            return x
        #print('pap=',pap)
        alpha = np.dot(r,r)/pap   #直接套用公式
        x1 = x + alpha*p
        r1 = r-alpha*np.dot(A,p)
        beta = np.dot(r1,r1)/np.dot(r,r)
        p1 = r1 +beta*p
        r = r1
        x = x1
        p = p1
        i=i+1
    return x

b = np.zeros(10)
b[1] = 1
print(b[1])
print(b.shape)