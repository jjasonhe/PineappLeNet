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
arrays are (N, Y, X, 2)
slices are (Y, X, 2)
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
	slice1 = slice1.transpose(1,0,2)
	slice2 = slice2.transpose(1,0,2)
	plt.figure()
	a.plt.subplot(2,1,1)
	plt.imshow(np.abs(slice1[:,:,1]), cmap='gray')
	a.set_title("t")
	a.plt.subplot(2,1,2)
	plt.imshow(np.abs(slice2[:,:,1]), cmap='gray')
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
Creates shapes.txt (if it doesn't exist), returns maxX, maxY, maxT for zero padding

i.create_h5(train_dict, val_dict, test_dict, maxX, maxY, maxT)
Creates samples.h5, 46.74 GB file containing data split into groups
'''
#num_train, num_val, num_test = 70, 10, 20

# train_data,val_data,test_data = i.create_dicts(num_train, num_val, num_test)
# maxX,maxY,maxT = i.init()										 # only call once
# i.create_h5(num_train, num_val, num_test, train_data, val_data, test_data, maxX, maxY, maxT) # only call once

'''
Test reading from hdf5 file, samples.h5
'''
f = h5.File('samples.h5', 'r')
print(f["train_curr"].shape)
plot_time_step(f["train_curr"][510], f["train_next"][510])
print(f["train_next"].shape)
print(f["val_curr"].shape)
print(f["val_next"].shape)
print(f["test_curr"].shape)
print(f["test_next"].shape)
f.close()

'''
Test plot functions
'''
# img = i.fetch(train_data[0],maxX,maxY,maxT)
# plot_time(img, 30, maxT)
# plot_slice(img, 0)
