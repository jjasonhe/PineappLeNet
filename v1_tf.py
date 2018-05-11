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
import h5py as h5 

'''
image shape is (Z, X, Y, T)
'''

'''
Visualization function for plotting each time frame
'''
def plot_time(data, Z, maxT):
	plt.figure()
	for j in range(0, maxT, 1):
		a = plt.subplot(3,6,j+1)
		plt.imshow(np.abs(img[Z,:,:,j]), cmap='gray')
		a.set_title("t = %d" % j)
	plt.show()

'''
Visualization function for plotting across the Z dimension
'''
def plot_slice(data, time):
	plt.figure()
	for j in range(0, 18, 1):
		a = plt.subplot(3,6,j+1)
		plt.imshow(np.abs(img[j*data.shape[0]//18,:,:,time]), cmap='gray')
		a.set_title("z = %d" % (j*data.shape[0]//18))
	plt.show()

'''
Creates hdf5 file, samples.h5
group /train contains 5362 datasets
group /val   contains  750 datasets
group /test  contains 1434 datasets

i.create_dicts()
Returns lists for train, val, test containing names of datasets (names)

i.init()
Creates shapes.txt (if it doesn't exist), returns maxX, maxY, maxT for zero padding

i.create_h5(train_dict, val_dict, test_dict, maxX, maxY, maxT)
Creates samples.h5, 46.74 GB file containing data split into groups
'''
num_train, num_val, num_test = 70, 10, 20

train_data,val_data,test_data = i.create_dicts(num_train, num_val, num_test)
maxX,maxY,maxT = i.init()										 # only call once
i.create_h5(num_train, num_val, num_test, train_data, val_data, test_data, maxX, maxY, maxT) # only call once

'''
Test reading from hdf5 file, samples.h5
'''
# f = h5.File('samples.h5', 'r')
# train_d = f["train"]
# val_d= f["val"]
# test_d = f["test"]
# print(train_d["%s_000" % train_data[0]].shape)
# print(val_d["%s_000" % val_data[0]].shape)
# print(test_d["%s_000" % test_data[0]].shape)

'''
Test plot functions
'''
# img = i.fetch(train_data[0],maxX,maxY,maxT)
# plot_time(img, 30, maxT)
# plot_slice(img, 0)
