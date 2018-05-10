import os
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sb
import importdata as i
import cfl
import warnings

#img.shape = (Z, X, Y, T)

def plot_time(data, Z, maxT):
	plt.figure()
	for j in range(0, maxT, 1):
		a = plt.subplot(3,6,j+1)
		plt.imshow(np.abs(img[Z,:,:,j]), cmap='gray')
		a.set_title("t = %d" % j)
	plt.show()

def plot_slice(data, time):
	plt.figure()
	for j in range(0, 18, 1):
		a = plt.subplot(3,6,j+1)
		plt.imshow(np.abs(img[j*data.shape[0]//18,:,:,time]), cmap='gray')
		a.set_title("z = %d" % (j*data.shape[0]//18))
	plt.show()

maxX,maxY,maxT = i.init()
train_data,val_data,test_data = i.create_dicts()
img = i.fetch(train_data[0],maxX,maxY,maxT)
# plot_time(img, 30, maxT)
# plot_slice(img, 0)