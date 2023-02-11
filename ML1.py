import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([np.arange(-10, 10, 1, dtype=np.float64)]).T
y = np.array([x ** 2 + 5]).T
one = np.ones(np.shape(x))
xsquare = np.square(x)
Xbar = np.hstack((one, x, xsquare))
W = np.array([4.94841015e+00, 5.25152244e-04, 1.00086308e+00], dtype=np.float64).reshape(-1, 1)
lr = 1.0e-7
for i in range(2):
    ypre = Xbar @ W
    Loss = np.mean((ypre - y) ** 2)
    dw0 = 2 * np.mean((ypre - y))
    dwx = 2 / np.size(x) * np.dot(Xbar[:, 1:].T,(ypre - y))
    W[0, 0] -= dw0 * lr
    W[1, 0] -= dwx[0] * lr
    W[2, 0] -= dwx[1] * lr
    print(W)
    print(Loss)
a=Xbar@W
plt.plot(x,a,color='#58b970', label='Regression Line')
plt.scatter(x, y, c='#ef5423', label='data points')
plt.show()
