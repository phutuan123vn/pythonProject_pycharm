import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([[1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]], dtype=np.float64).T
y = np.array([[245, 312, 279, 308, 199, 219, 405, 324, 319, 255]], dtype=np.float64).T
x=x/np.max(x)
ymax=np.max(y)
y=y/ymax
xbar = np.hstack((np.ones(x.shape), x))
A = xbar.T @ xbar
W = np.array([[95.00001711,0.11156574]],dtype=np.float64)
LR = LinearRegression()
LR = LR.fit(x, y)
a = LR.coef_
b = LR.intercept_
wtest=np.array([b,a])
lr = 1.0e-08
for i in range(10000):
    ypre=np.matmul(xbar,W.T)
    Loss=np.mean((ypre-y)**2)
    dw0=np.mean((ypre-y))
    dwx=np.matmul(x.T,(ypre-y))
    W[0]-=lr*dw0
    W[0,1]-=lr*dwx
    print(Loss)
    print(dwx)
print(W)
plt.scatter(x,y)
plt.plot(x, xbar @ W.T, '-',color='red',label='W train')
plt.plot(x, xbar @ wtest, '--',color='blue',label='Wtest')
plt.legend()
plt.show()
