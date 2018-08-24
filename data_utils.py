from __future__ import print_function, division
import os
import ast
from operator import itemgetter
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import cfl
import h5py as h5
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

plt.ion()

"""
Data Pipeline from v1:

init
- Check for shapes.txt
- Create shapes.txt
	- cfl.read to import image
	- np.squeeze to remove extra dimensions
	- Write shapes to shapes.txt
- Open shapes.txt
- Reads shapes.txt into a string
- list(ast.literal_eval(shp)) turns string into list
- Returns maxX, maxY, maxT

create_dicts
- Splits data into train, val, test
- Returns list of folder names for each data category

fetch
- cfl.read to import patient
- np.squeeze to remove extra dimensions
- Takes only first channel
- Zero-pad in X and Y directions
- Rearrange dimensions into (Z, T, Y, X) # This should've been X, Y...
- Flip in X and flip in Y
"""

"""
Desired Pipeline for v2:
x- Find maxX, maxY, maxT
x- Be able to customize how many sets for train, val, test
x- Import a patient
x- Remove extra dimensions
x- Take only first channel
x- Rearrange into (Z, T, X, Y)
x- Zero-pad in X and Y directions
x- Flip in X and flip in Y
x- Add extra axis for C
x- Take absolute value of image
x- Normalize (divide by max of that patient)
?- Prune out the low contrast change slices from each patient

Plan 1:
- Do all this preprocessing^ in a separate function (data_prep)
- Create 3 h5 files that have already been preprocessed
- That way, just create 3 DCEDatasets (train, val, test)
- Con: modifying my preprocessing routine (such as pruning) will require recreation of h5 files (time-consuming)
- But most of the preprocessing shouldn't change? Besides pruning?
"""

def data_prep(num_train=70, num_val=10, num_test=20):
	txt = open("patients.txt", "r")
	patients = txt.read()
	txt.close()
	patients = list(ast.literal_eval(patients))

	train_dict = patients[0:num_train]
	val_dict = patients[num_train:num_train + num_val]
	test_dict = patients[num_train + num_val:num_train + num_val + num_test]

	if (not os.path.exists("shapes.txt")) or (not os.path.getsize("shapes.txt") > 0):
		txt = open("shapes.txt", "w")
		i = 0
		for f in patients:
			img = cfl.read('datasets/%s/im_dce' % f)
			img = np.squeeze(img)
			txt.write("%s," % (img.shape,))
			print(i)
			i = i + 1
		txt.close()
	txt = open("shapes.txt", "r")
	shp = txt.read()
	txt.close()
	shp = list(ast.literal_eval(shp))
	maxX = max(shp, key=itemgetter(4))[4]
	maxY = max(shp, key=itemgetter(3))[3]
	maxT = max(shp, key=itemgetter(0))[0]

	hf = h5.File('preproc_data.h5', 'a')
	hf.create_dataset('train', (0, maxT, maxX, maxY, 1), maxshape=(None, maxT, maxX, maxY, 1))
	hf.create_dataset('val', (0, maxT, maxX, maxY, 1), maxshape=(None, maxT, maxX, maxY, 1))
	hf.create_dataset('test', (0, maxT, maxX, maxY, 1), maxshape=(None, maxT, maxX, maxY, 1))

	for patient in train_dict:
		img = cfl.read('datasets/%s/im_dce' % patient)
		img = np.squeeze(img)
		img = img[:,0,:,:,:]
		img = np.transpose(img, (1, 0, 3, 2)) # (Z, T, X, Y)
		img = img[:,:,::-1,::-1]
		if img.shape[2] < maxX:
			img = np.pad(img, ((0,),(0,),((maxX-img.shape[2])//2,),(0,)), mode='constant')
		if img.shape[3] < maxY:
			img = np.pad(img, ((0,),(0,),(0,),((maxY-img.shape[3])//2,)), mode='constant')
		img = img[:,:,:,:,np.newaxis]
		img = np.absolute(img).astype(np.float32)
		max_img = np.amax(img, axis=(1,2,3))
		img = img/max_img[:,np.newaxis,np.newaxis,np.newaxis,:]
		print(img.shape)
		hf['train'].resize((hf['train'].shape[0] + img.shape[0]), axis=0)
		hf['train'][-img.shape[0]:] = img
		print(hf['train'].shape)

	for patient in val_dict:
		img = cfl.read('datasets/%s/im_dce' % patient)
		img = np.squeeze(img)
		img = img[:,0,:,:,:]
		img = np.transpose(img, (1, 0, 3, 2)) # (Z, T, X, Y)
		img = img[:,:,::-1,::-1]
		if img.shape[2] < maxX:
			img = np.pad(img, ((0,),(0,),((maxX-img.shape[2])//2,),(0,)), mode='constant')
		if img.shape[3] < maxY:
			img = np.pad(img, ((0,),(0,),(0,),((maxY-img.shape[3])//2,)), mode='constant')
		img = img[:,:,:,:,np.newaxis]
		img = np.absolute(img).astype(np.float32)
		max_img = np.amax(img, axis=(1,2,3))
		img = img/max_img[:,np.newaxis,np.newaxis,np.newaxis,:]
		print(img.shape)
		hf['val'].resize((hf['val'].shape[0] + img.shape[0]), axis=0)
		hf['val'][-img.shape[0]:] = img
		print(hf['val'].shape)

	for patient in test_dict:
		img = cfl.read('datasets/%s/im_dce' % patient)
		img = np.squeeze(img)
		img = img[:,0,:,:,:]
		img = np.transpose(img, (1, 0, 3, 2)) # (Z, T, X, Y)
		img = img[:,:,::-1,::-1]
		if img.shape[2] < maxX:
			img = np.pad(img, ((0,),(0,),((maxX-img.shape[2])//2,),(0,)), mode='constant')
		if img.shape[3] < maxY:
			img = np.pad(img, ((0,),(0,),(0,),((maxY-img.shape[3])//2,)), mode='constant')
		img = img[:,:,:,:,np.newaxis]
		img = np.absolute(img).astype(np.float32)
		max_img = np.amax(img, axis=(1,2,3))
		img = img/max_img[:,np.newaxis,np.newaxis,np.newaxis,:]
		print(img.shape)
		hf['test'].resize((hf['test'].shape[0] + img.shape[0]), axis=0)
		hf['test'][-img.shape[0]:] = img
		print(hf['test'].shape)

	hf.close()



class DCEDataset(Dataset):
	def __init__(self, mode, path, transform=None):
		hf = h5.File(path, 'r')
		self.mode = mode
		self.path = path
		self.len = hf[mode].shape[0]
		hf.close()

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		hf = h5.File(self.path, 'r')
		ds = hf[self.mode]
		return ds[idx]
