import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.cvtColor(cv2.imread('avatar.jpg'), cv2.COLOR_BGR2GRAY)
am_ban = 255 - img
fig = plt.figure(figsize=(16, 9))
(anh1, anh2), (hist1, hist2) = fig.subplots(2, 2)
hist, bin_edge = np.histogram(img, bins=np.arange(257))
xs = hist / sum(hist)
anh1.hist(img.ravel(),256,[0,256])
anh1.plot(np.arange(256), xs.cumsum()*3.0e+04, color='red')

plt.show()

