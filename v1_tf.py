import os
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sb
import importdata as i
import cfl
import h5py as h5 

'''
arrays are (N, Y, X)
slices are (Y, X)
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


def plot_time_step(slice1, slice2):
	slice1 = slice1.transpose(1,0)
	slice2 = slice2.transpose(1,0)
	plt.figure()
	a = plt.subplot(2,1,1)
	plt.imshow(np.abs(slice1[:,:]), cmap='gray')
	a.set_title("t")
	a = plt.subplot(2,1,2)
	plt.imshow(np.abs(slice2[:,:]), cmap='gray')
	a.set_title("t+1")
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
/train_curr 91154 /train_next 91154
/val_curr   12750 /val_next   12750
/test_curr  24378 /test_next  24378

i.create_dicts()
Returns lists for train, val, test containing names of datasets (names)

i.init()
Creates shapes.txt (if it doesn't exist); returns maxX, maxY, maxT for zero padding

i.<dict>_h5(<dict>, maxX, maxY, maxT)
Each call creates 2 h5 files, current and next
'''
num_train, num_val, num_test = 70, 10, 20

train_data,val_data,test_data = i.create_dicts(num_train, num_val, num_test)
# maxX,maxY,maxT = i.init() # call once
# i.create_h5(train_data, val_data, test_data, maxX, maxY, maxT) # call once
# i.val_h5(val_data, maxX, maxY, maxT)
# i.test_h5(test_data, maxX, maxY, maxT)
# i.train_h5(train_data, maxX, maxY, maxT)

'''
Test reading from hdf5 file, samples.h5
'''
fc = h5.File('train_curr.h5', 'r')
fn = h5.File('train_next.h5', 'r')
print(fc[train_data[0]].shape)
print(fn[train_data[0]].shape)
plot_time_step(fc[train_data[0]][510], fn[train_data[0]][510])
# f = h5.File('samples.h5', 'r')
# print(f["train_curr"].shape)
# plot_time_step(f["train_curr"][510], f["train_next"][510])
# print(f["train_next"].shape)
# print(f["val_curr"].shape)
# print(f["val_next"].shape)
# print(f["test_curr"].shape)
# print(f["test_next"].shape)
# f.close()
