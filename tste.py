import numpy as np
import pandas as pd

data = pd.read_csv('Chapter 4 - Example - Data.csv')
data = data.astype(int)
x = np.array([data],dtype=int).reshape(-1,1)


def behonT(img, T):
    lst = []
    for i in img:
        if i < T:
            lst.append(i)
    return lst


def lonhonT(img, T):
    lst = []
    for i in img:
        if i > T:
            lst.append(i)
    return lst


x1 = np.array([behonT(x, 127)])
x2 = np.array([lonhonT(x, 127)])
while True:
    deltaT=(x1.mean()+x2.mean())/2
    x1 = np.array([behonT(x, deltaT)])
    x2 = np.array([lonhonT(x, deltaT)])
    if deltaT
    print(deltaT)
