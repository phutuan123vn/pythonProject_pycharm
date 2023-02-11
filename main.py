import numpy as np


def yp(x, w):
    # print(np.dot(x,w))
    return np.dot(x, w)


def JCal(yp):
    return ((yp - Y) ** 2).mean()


def gradient(yp):
    a = np.mean(2*X[:,1]*(ypre-Y))
    print(a)
    return a


x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])
n = np.size(x)
X = np.ones([n, 2])
Y = np.array(y).reshape(-1, 1)
W = np.array([0, 1], dtype=float).reshape([2, 1])
X[:, 1] = x
X[:, 0] = 1
lr = 0.001
print(np.dot(W, 3))
print(Y)
for i in range(100):
    ypre = yp(X, W)
    J = JCal(ypre)
    dw = gradient(ypre)
    W -= dw * lr
    print(ypre)
print(f"Predict f(9): {9 * W[1] + W[0]}")
print(f"Predict f(6): {5 * W[1] + W[0]}")
print(W, J)
# # def ypred(x):
# #     return np.dot(x,W)
# #
# #
# # def loss(yp, y):
# #     a = np.subtract(yp, y)
# #     a = a * 1 / 2
# #     return np.mean(np.power(a, 2))
# #
# #
# # def graient(x, yp, y):
# #     return np.dot(x, yp - y).mean()
# #
# #
# # X[:, 1] = x
# # X[:, 0] = 1
# # lr = 0.01
# # for i in range(100):
# #     yp = ypred(X)
# #     print(yp)
# #     J = loss(yp, y)
# #     dw = graient(X, yp, y)
# #     W -= lr * dw
# # print(f"PREDICT f(9): {ypred(9):.3f}")
