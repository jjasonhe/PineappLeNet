import os
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sb
import importdata
import cfl
import warnings

#data.shape = (Z, X, Y, T)

data = importdata.i()
print(data.shape)
plt.figure()
plt.subplot(221)
plt.imshow(np.abs(data[30,:,:,0]), cmap='gray')
plt.subplot(222)
plt.imshow(np.abs(data[30,:,:,5]), cmap='gray')
plt.subplot(223)
plt.imshow(np.abs(data[30,:,:,10]), cmap='gray')
plt.subplot(224)
plt.imshow(np.abs(data[30,:,:,15]), cmap='gray')
plt.show()