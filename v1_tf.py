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

# img.shape = (Z, X, Y, T)

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

# Creates hdf5 file, samples.h5
# KeysView(<HDF5 group "/train" (5362 members)>)
# KeysView(<HDF5 group "/val" (750 members)>)
# KeysView(<HDF5 group "/test" (1434 members)>)
# maxX,maxY,maxT = i.init()
train_data,val_data,test_data = i.create_dicts()
# i.create_h5(train_data, val_data, test_data, maxX, maxY, maxT)

f = h5.File('samples.h5', 'r')
train_d = f["train"]
val_d= f["val"]
test_d = f["test"]
print(train_d["%s_000" % train_data[0]].shape)
print(val_d["%s_000" % val_data[0]].shape)
print(test_d["%s_000" % test_data[0]].shape)

# img = i.fetch(train_data[0],maxX,maxY,maxT)
# plot_time(img, 30, maxT)
# plot_slice(img, 0)

# f = h5.File('test.h5', 'w')
# grp = f.create_group("train")
# dset = grp.create_dataset("foldername", data=img)
# print(grp.keys())

#for i in range()