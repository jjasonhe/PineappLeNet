import os
import ast
import numpy as np
import cfl
import h5py as h5
import tensorflow as tf
import random
from operator import itemgetter

folders = [
"01Apr16_Ex18785_Ser6",
"01Aug16_Ex19670_Ser2",
"01Dec15_Ex17585_Ser9",
"01Dec16_Ex20906_Ser15",
"01Dec16_Ex4177_Ser13",
"01Feb16_Ex18149_Ser7",
"01Feb16_Ex2749_Ser10",
"01Jul16_Ex19381_Ser14",
"01Jun15_Ex15796_Ser8",
"01Jun15_Ex15797_Ser8",
"01Jun15_Ex1732_Ser7",
"01Jun16_Ex19071_Ser16",
"01Mar16_Ex18458_Ser7",
"01May15_Ex15519_Ser7",
"01Nov16_Ex4031_Ser17",
"01Sep16_Ex19997_Ser12",
"01Sep16_Ex19999_Ser13",
"02Aug16_Ex19685_Ser10",
"02Aug16_Ex3590_Ser17",
"02Dec15_Ex17598_Ser16",
"02Dec16_Ex20912_Ser6",
"02Dec16_Ex20914_Ser9",
"02Dec16_Ex20917_Ser11",
"02Feb15_Ex14663_Ser6",
"02Jan15_Ex14374_Ser4",
"02Jan15_Ex14376_Ser6",
"02Jan15_Ex14378_Ser14",
"02Jan15_Ex14378_Ser17",
"02Jul15_Ex16112_Ser8",
"02Jun15_Ex15816_Ser11",
"02Mar15_Ex14925_Ser8",
"02Mar16_Ex18464_Ser12",
"02Mar16_Ex18469_Ser11",
"02Nov15_Ex17305_Ser7",
"02Nov15_Ex17309_Ser7",
"02Oct15_Ex17012_Ser3",
"02Sep16_Ex20004_Ser11",
"03Apr15_Ex15260_Ser9",
"03Apr15_Ex15261_Ser9",
"03Apr16_Ex18799_Ser8",
"03Aug16_Ex19696_Ser18",
"03Dec15_Ex17610_Ser7",
"03Dec16_Ex20920_Ser9",
"03Feb15_Ex14673_Ser11",
"03Feb15_Ex14673_Ser8",
"03Jun15_Ex15824_Ser12",
"03Jun15_Ex1745_Ser6",
"03Jun16_Ex19093_Ser8",
"03Jun16_Ex19094_Ser16",
"03Jun16_Ex3303_Ser10",
"03Mar15_Ex14940_Ser4",
"03Mar16_Ex18481_Ser9",
"03May16_Ex3175_Ser12",
"03May16_Ex3179_Ser10",
"03Nov15_Ex17321_Ser11",
"03Nov15_Ex17322_Ser7",
"03Nov16_Ex20627_Ser10",
"03Nov16_Ex20627_Ser9",
"03Nov16_Ex20633_Ser15",
"03Nov16_Ex20634_Ser15",
"03Oct16_Ex20309_Ser9",
"03Oct16_Ex20316_Ser11",
"03Oct16_Ex3893_Ser12",
"03Sep15_Ex16737_Ser8",
"03Sep15_Ex16741_Ser7",
"04Apr16_Ex18807_Ser13",
"04Apr16_Ex18810_Ser8",
"04Apr16_Ex18812_Ser12",
"04Dec15_Ex17618_Ser9",
"04Dec15_Ex17619_Ser6",
"04Dec15_Ex17624_Ser12",
"04Feb15_Ex14688_Ser10",
"04Jan16_Ex17902_Ser8",
"04Jan16_Ex17904_Ser9",
"04Jan16_Ex17905_Ser14",
"04Mar16_Ex18488_Ser7",
"04Mar16_Ex18492_Ser7",
"04May15_Ex15536_Ser9",
"04May15_Ex15538_Ser9",
"04May15_Ex15540_Ser11",
"04Nov15_Ex17333_Ser8",
"04Nov16_Ex20640_Ser9",
"04Nov16_Ex20643_Ser9",
"04Nov16_Ex20646_Ser14",
"04Nov16_Ex20648_Ser12",
"04Oct16_Ex20322_Ser10",
"04Sep15_Ex16747_Ser6",
"04Sep15_Ex16747_Ser7",
"04Sep15_Ex16749_Ser10",
"04Sep15_Ex16750_Ser9",
"05Aug15_Ex2041_Ser6",
"05Aug16_Ex19719_Ser15",
"05Dec16_Ex20930_Ser11",
"05Dec16_Ex4196_Ser14",
"05Feb15_Ex14696_Ser9",
"05Feb16_Ex18191_Ser8",
"05Feb16_Ex18196_Ser12",
"05Feb16_Ex18196_Ser8",
"05Jan15_Ex14391_Ser7",
"05Jan16_Ex17914_Ser7"
]

def init():
	if (not os.path.exists("shapes.txt")) or (not os.path.getsize("shapes.txt") > 0):
		txt = open("shapes.txt", "w")
		i = 0
		for f in folders:
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
	return maxX,maxY,maxT

def create_dicts(num_train, num_val, num_test):
	train_dict = list(folders[0:num_train])
	val_dict = list(folders[num_train:num_train+num_val])
	test_dict = list(folders[num_train+num_val:num_train+num_val+num_test])
	return train_dict,val_dict,test_dict

def fetch(f, maxX, maxY, maxT):
	img = cfl.read('datasets/%s/im_dce' % f)
	img = np.squeeze(img)
	# img.shape is (T, C, Z, Y, X)
	img = img[:,0,:,:,:]
	# img.shape is (T, Z, Y, X)
	if img.shape[3] < maxX:
		img = np.pad(img, ((0,),(0,),(0,),((maxX-img.shape[3])//2,)), mode='constant')
	if img.shape[2] < maxY:
		img = np.pad(img, ((0,),(0,),((maxY-img.shape[2])//2,),(0,)), mode='constant')
	img = np.transpose(img, (1, 0, 2, 3)) # (Z, T, Y, X)
	img = img[:,:,::-1,::-1]
	return img

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def train_h5(train_dict, maxX, maxY, maxT):
	f = h5.File('datasets/train.h5', 'w')
	i = 0
	for patient in train_dict:
		img = fetch(patient, maxX, maxY, maxT)
		Z,T,Y,X = img.shape
		img = img[:,:,:,:,np.newaxis]
		f.create_dataset("%s" % patient, data=img)
		print("trained patient %d" % i)
		i = i + 1
	f.close()
	return
        
def val_h5(val_dict, maxX, maxY, maxT):
	f = h5.File('datasets/val.h5', 'w')
	i = 0
	for patient in val_dict:
		img = fetch(patient, maxX, maxY, maxT)
		Z,T,Y,X = img.shape
		img = img[:,:,:,:,np.newaxis]
		f.create_dataset("%s" % patient, data=img)
		print("validated patient %d" % i)
		i = i + 1
	f.close()
	return

def test_h5(test_dict, maxX, maxY, maxT):
	f = h5.File('datasets/test.h5', 'w')
	i = 0
	for patient in test_dict:
		img = fetch(patient, maxX, maxY, maxT)
		Z,T,Y,X = img.shape
		img = img[:,:,:,:,np.newaxis]
		f.create_dataset("%s" % patient, data=img)
		print("tested patient %d" % i)
		i = i + 1
	f.close()
	return

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# PRODUCES TIMESERIES

def what_tf(mode, patients):
    if os.path.exists('datasets/{}.tfrecords'.format(mode)):
        print('{}.tfrecords already exists'.format(mode))
        return
    path_h5 = 'datasets/{}.h5'.format(mode)
    f = h5.File(path_h5, 'r')
    path_tf = 'datasets/{}.tfrecords'.format(mode)
    writer = tf.python_io.TFRecordWriter(path_tf)
    for i,p in enumerate(patients):
        imgs = f[p]
        imgs = np.absolute(imgs).astype(np.float32)
        max_imgs = np.amax(imgs, axis=(1,2,3))
        imgs = imgs/max_imgs[:,np.newaxis,np.newaxis,np.newaxis,:]
        for z in range(f[p].shape[0]):
            img = imgs[z,:,:,:,:]
            example = tf.train.Example(features=tf.train.Features(feature={
                'img': _bytes_feature(img.tostring())
            }))
            writer.write(example.SerializeToString())
        print('patient {}'.format(i))
    writer.close()
    return

def create_dataset(mode, patients):
    filenames = 'datasets/{}.tfrecords'.format(mode)
    dataset = tf.data.TFRecordDataset(filenames)
    def _prep_tfrecord(example):
        features = tf.parse_single_example(
            example,
            features={
                'img': tf.FixedLenFeature([], tf.string)
            }
        )
        img_record_bytes = tf.decode_raw(features['img'], tf.float32)
        img = tf.reshape(img_record_bytes, [18, 224, 192, 1])
        img = tf.transpose(img, [0, 2, 1, 3])
        features = img[:-1,:,:,:]
        labels = img[1:,:,:,:]
        return features, labels
    dataset = dataset.map(_prep_tfrecord)
    return dataset

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# ONE-TO-ONE

def what_tf_fcn8(mode, patients):
    if os.path.exists('datasets/{}_fcn8.tfrecords'.format(mode)):
        print('{}_fcn8.tfrecords already exists'.format(mode))
        return
    num_ex = 0
    path_h5 = 'datasets/{}.h5'.format(mode)
    f = h5.File(path_h5, 'r')
    path_tf = 'datasets/{}_fcn8.tfrecords'.format(mode)
    writer = tf.python_io.TFRecordWriter(path_tf)
    for i,p in enumerate(patients):
        imgs = f[p]
        imgs = np.absolute(imgs).astype(np.float32)
        max_imgs = np.amax(imgs, axis=(1,2,3))
        imgs = imgs/max_imgs[:,np.newaxis,np.newaxis,np.newaxis,:]
        for z in range(f[p].shape[0]):
            for t in range(f[p].shape[1]-1):
                img_curr = imgs[z,t,:,:,:]
                img_next = imgs[z,t+1,:,:,:]
                example = tf.train.Example(features=tf.train.Features(feature={
                    'img_curr': _bytes_feature(img_curr.tostring()),
                    'img_next': _bytes_feature(img_next.tostring())
                }))
                writer.write(example.SerializeToString())
                num_ex += 1
                print('example {}'.format(num_ex))
        print('patient {}'.format(i))
    writer.close()
    return

def create_dataset_fcn8(mode, patients):
    filenames = 'datasets/{}_fcn8.tfrecords'.format(mode)
    dataset = tf.data.TFRecordDataset(filenames)
    def _prep_tfrecord(example):
        features = tf.parse_single_example(
            example,
            features={'img_curr': tf.FixedLenFeature([], tf.string),
                      'img_next': tf.FixedLenFeature([], tf.string)}
        )
        curr_record_bytes = tf.decode_raw(features['img_curr'], tf.float32)
        next_record_bytes = tf.decode_raw(features['img_next'], tf.float32)
        features = tf.reshape(curr_record_bytes, [224, 192, 1])
        features = tf.transpose(features, [1, 0, 2])
        labels = tf.reshape(next_record_bytes, [224, 192, 1])
        labels = tf.transpose(labels, [1, 0, 2])
        return features, labels
    dataset = dataset.map(_prep_tfrecord)
    return dataset

def create_dataset_diff(mode, patients):
    filenames = 'datasets/{}_fcn8.tfrecords'.format(mode)
    dataset = tf.data.TFRecordDataset(filenames)
    def _prep_tfrecord(example):
        features = tf.parse_single_example(
            example,
            features={'img_curr': tf.FixedLenFeature([], tf.string),
                      'img_next': tf.FixedLenFeature([], tf.string)}
        )
        curr_record_bytes = tf.decode_raw(features['img_curr'], tf.float32)
        next_record_bytes = tf.decode_raw(features['img_next'], tf.float32)
        features = tf.reshape(curr_record_bytes, [224, 192, 1])
        features = tf.transpose(features, [1, 0, 2])
        next_img = tf.reshape(next_record_bytes, [224, 192, 1])
        next_img = tf.transpose(next_img, [1, 0, 2])
        labels = next_img - features
        return features, labels
    dataset = dataset.map(_prep_tfrecord)
    return dataset

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# PRUNED TIMESERIES

def what_tf_pruned(mode, patients):
    if os.path.exists('datasets/{}_p.tfrecords'.format(mode)):
        print('{}_p.tfrecords already exists'.format(mode))
        return
    path_h5 = 'datasets/{}.h5'.format(mode)
    f = h5.File(path_h5, 'r')
    path_tf = 'datasets/{}_p.tfrecords'.format(mode)
    writer = tf.python_io.TFRecordWriter(path_tf)
    for i,p in enumerate(patients):
        imgs = f[p]
        imgs = np.transpose(imgs, (0,1,3,2,4))
        imgs = np.absolute(imgs).astype(np.float32)
        avg_diffs = np.zeros((imgs.shape[0]))
        for z in range(imgs.shape[0]):
            timeseries = imgs[z,:,:,:,:]
            diffs = np.zeros((timeseries.shape[0]-1, timeseries.shape[1], timeseries.shape[2], timeseries.shape[3]))
            for t in range(timeseries.shape[0]-1):
                diffs[t,:,:,:] = np.absolute(timeseries[t+1,:,:,:] - timeseries[t,:,:,:])
            avg_diffs[z] = np.mean(diffs)
        
        z_pruned = np.argsort(-avg_diffs) # sort in descending order
        max_imgs = np.amax(imgs, axis=(1,2,3))
        imgs = imgs/max_imgs[:,np.newaxis,np.newaxis,np.newaxis,:]
        for z in z_pruned[:20]:
            img = imgs[z,:,:,:,:]
            example = tf.train.Example(features=tf.train.Features(feature={
                'img': _bytes_feature(img.tostring())
            }))
            writer.write(example.SerializeToString())
        print('patient {}'.format(i))
    writer.close()
    return

def create_dataset_p(mode, patients):
    filenames = 'datasets/{}_p.tfrecords'.format(mode)
    dataset = tf.data.TFRecordDataset(filenames)
    def _prep_tfrecord(example):
        features = tf.parse_single_example(
            example,
            features={
                'img': tf.FixedLenFeature([], tf.string)
            }
        )
        img_record_bytes = tf.decode_raw(features['img'], tf.float32)
        img = tf.reshape(img_record_bytes, [18, 192, 224, 1])
        features = img[:-1,:,:,:]
        labels = img[1:,:,:,:]
        return features, labels
    dataset = dataset.map(_prep_tfrecord)
    return dataset

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# TEA5POON IS THREE-TO-ONE TIMESERIES

def what_tf_tea5poon(mode, patients):
    if os.path.exists('datasets/{}_t5.tfrecords'.format(mode)):
        print('{}_t5.tfrecords already exists'.format(mode))
        return
    num_ex = 0
    path_h5 = 'datasets/{}.h5'.format(mode)
    f = h5.File(path_h5, 'r')
    path_tf = 'datasets/{}_t5.tfrecords'.format(mode)
    writer = tf.python_io.TFRecordWriter(path_tf)
    for i,p in enumerate(patients):
        imgs = f[p]
        imgs = np.transpose(imgs, (0,1,3,2,4))
        imgs = np.absolute(imgs).astype(np.float32)
        avg_diffs = np.zeros((imgs.shape[0]))
        for z in range(imgs.shape[0]):
            timeseries = imgs[z,:,:,:,:]
            diffs = np.zeros((timeseries.shape[0]-1, timeseries.shape[1], timeseries.shape[2], timeseries.shape[3]))
            for t in range(timeseries.shape[0]-1):
                diffs[t,:,:,:] = np.absolute(timeseries[t+1,:,:,:] - timeseries[t,:,:,:])
            avg_diffs[z] = np.mean(diffs)
        z_pruned = np.argsort(-avg_diffs) # sort in descending order
        max_imgs = np.amax(imgs, axis=(1,2,3))
        imgs = imgs/max_imgs[:,np.newaxis,np.newaxis,np.newaxis,:]
        for z in z_pruned[:20]:
            # imgs[z,:,:,:,:] is a timeseries
            # I want to take the time series from 0 to 14, 1 to 15, 2 to 16 and concatenate
            # Then for features, I want 3 through 17
            img_c = np.stack((imgs[z,0:15,:,:,:],imgs[z,1:16,:,:,:],imgs[z,2:17,:,:,:]), axis=3) # (15,192,224,3)
            img_c = np.squeeze(img_c)
            img_n = imgs[z,3:18,:,:,:] # (1,15,192,224,1)
            example = tf.train.Example(features=tf.train.Features(feature={
                'img_curr': _bytes_feature(img_c.tostring()),
                'img_next': _bytes_feature(img_n.tostring())
            }))
            writer.write(example.SerializeToString())
            num_ex += 1
        print('patient {}'.format(i))
        print('example {}'.format(num_ex))
    writer.close()
    return

def create_dataset_tea5poon(mode, patients):
    filenames = 'datasets/{}_t5.tfrecords'.format(mode)
    dataset = tf.data.TFRecordDataset(filenames)
    def _prep_tfrecord(example):
        features = tf.parse_single_example(
            example,
            features={'img_curr': tf.FixedLenFeature([], tf.string),
                      'img_next': tf.FixedLenFeature([], tf.string)}
        )
        curr_record_bytes = tf.decode_raw(features['img_curr'], tf.float32)
        next_record_bytes = tf.decode_raw(features['img_next'], tf.float32)
        features = tf.reshape(curr_record_bytes, [15, 192, 224, 3])
        labels = tf.reshape(next_record_bytes, [15, 192, 224, 1])
        return features, labels
    dataset = dataset.map(_prep_tfrecord)
    return dataset

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# THREE-TO-ONE

def what_tf_three(mode, patients):
    if os.path.exists('datasets/{}_rip.tfrecords'.format(mode)):
        print('{}_rip.tfrecords already exists'.format(mode))
        return
    num_ex = 0
    path_h5 = 'datasets/{}.h5'.format(mode)
    f = h5.File(path_h5, 'r')
    path_tf = 'datasets/{}_rip.tfrecords'.format(mode)
    writer = tf.python_io.TFRecordWriter(path_tf)
    for i,p in enumerate(patients):
        imgs = f[p]
        imgs = np.transpose(imgs, (0,1,3,2,4))
        imgs = np.absolute(imgs).astype(np.float32)
        avg_diffs = np.zeros((imgs.shape[0]))
        for z in range(imgs.shape[0]):
            timeseries = imgs[z,:,:,:,:]
            diffs = np.zeros((timeseries.shape[0]-1, timeseries.shape[1], timeseries.shape[2], timeseries.shape[3]))
            for t in range(timeseries.shape[0]-1):
                diffs[t,:,:,:] = np.absolute(timeseries[t+1,:,:,:] - timeseries[t,:,:,:])
            avg_diffs[z] = np.mean(diffs)
        z_pruned = np.argsort(-avg_diffs) # sort in descending order
        max_imgs = np.amax(imgs, axis=(1,2,3))
        imgs = imgs/max_imgs[:,np.newaxis,np.newaxis,np.newaxis,:]
        for z in z_pruned[:20]:
            for t in range(f[p].shape[1]-3):
                img_0 = imgs[z,t,:,:,:]
                img_1 = imgs[z,t+1,:,:,:]
                img_2 = imgs[z,t+2,:,:,:]
                img_n = imgs[z,t+3,:,:,:]
                img_c = np.stack((img_0, img_1, img_2), axis=2)
                img_c = np.squeeze(img_c)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'img_curr': _bytes_feature(img_c.tostring()),
                    'img_next': _bytes_feature(img_n.tostring())
                }))
                writer.write(example.SerializeToString())
                num_ex += 1
        print('patient {}'.format(i))
        print('example {}'.format(num_ex))
    writer.close()
    return

def create_dataset_three(mode, patients):
    filenames = 'datasets/{}_rip.tfrecords'.format(mode)
    dataset = tf.data.TFRecordDataset(filenames)
    def _prep_tfrecord(example):
        features = tf.parse_single_example(
            example,
            features={'img_curr': tf.FixedLenFeature([], tf.string),
                      'img_next': tf.FixedLenFeature([], tf.string)}
        )
        curr_record_bytes = tf.decode_raw(features['img_curr'], tf.float32)
        next_record_bytes = tf.decode_raw(features['img_next'], tf.float32)
        features = tf.reshape(curr_record_bytes, [192, 224, 3])
        labels = tf.reshape(next_record_bytes, [192, 224, 1])
        return features, labels
    dataset = dataset.map(_prep_tfrecord)
    return dataset

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# TWO-TO-ONE

def what_tf_two(mode, patients):
    if os.path.exists('datasets/{}_two.tfrecords'.format(mode)):
        print('{}_two.tfrecords already exists'.format(mode))
        return
    num_ex = 0
    path_h5 = 'datasets/{}.h5'.format(mode)
    f = h5.File(path_h5, 'r')
    path_tf = 'datasets/{}_two.tfrecords'.format(mode)
    writer = tf.python_io.TFRecordWriter(path_tf)
    for i,p in enumerate(patients):
        imgs = f[p]
        imgs = np.transpose(imgs, (0,1,3,2,4))
        imgs = np.absolute(imgs).astype(np.float32)
        avg_diffs = np.zeros((imgs.shape[0]))
        for z in range(imgs.shape[0]):
            timeseries = imgs[z,:,:,:,:]
            diffs = np.zeros((timeseries.shape[0]-1, timeseries.shape[1], timeseries.shape[2], timeseries.shape[3]))
            for t in range(timeseries.shape[0]-1):
                diffs[t,:,:,:] = np.absolute(timeseries[t+1,:,:,:] - timeseries[t,:,:,:])
            avg_diffs[z] = np.mean(diffs)
        z_pruned = np.argsort(-avg_diffs) # sort in descending order
        max_imgs = np.amax(imgs, axis=(1,2,3))
        imgs = imgs/max_imgs[:,np.newaxis,np.newaxis,np.newaxis,:]
        for z in z_pruned[:20]:
            for t in range(f[p].shape[1]-2):
                img_0 = imgs[z,t,:,:,:]
                img_1 = imgs[z,t+1,:,:,:]
                img_n = imgs[z,t+2,:,:,:]
                img_c = np.stack((img_0, img_1), axis=2)
                img_c = np.squeeze(img_c)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'img_curr': _bytes_feature(img_c.tostring()),
                    'img_next': _bytes_feature(img_n.tostring())
                }))
                writer.write(example.SerializeToString())
                num_ex += 1
        print('patient {}'.format(i))
        print('example {}'.format(num_ex))
    writer.close()
    return

def create_dataset_two(mode, patients):
    filenames = 'datasets/{}_two.tfrecords'.format(mode)
    dataset = tf.data.TFRecordDataset(filenames)
    def _prep_tfrecord(example):
        features = tf.parse_single_example(
            example,
            features={'img_curr': tf.FixedLenFeature([], tf.string),
                      'img_next': tf.FixedLenFeature([], tf.string)}
        )
        curr_record_bytes = tf.decode_raw(features['img_curr'], tf.float32)
        next_record_bytes = tf.decode_raw(features['img_next'], tf.float32)
        features = tf.reshape(curr_record_bytes, [192, 224, 2])
        labels = tf.reshape(next_record_bytes, [192, 224, 1])
        return features, labels
    dataset = dataset.map(_prep_tfrecord)
    return dataset

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# ROCKETS IN 7 LET'S GO

def what_tf_g7(mode, patients):
    if os.path.exists('datasets/{}_g7.tfrecords'.format(mode)):
        print('{}_g7.tfrecords already exists'.format(mode))
        return
    num_ex = 0
    path_h5 = 'datasets/{}.h5'.format(mode)
    f = h5.File(path_h5, 'r')
    path_tf = 'datasets/{}_g7.tfrecords'.format(mode)
    writer = tf.python_io.TFRecordWriter(path_tf)
    for i,p in enumerate(patients):
        imgs = f[p]
        imgs = np.transpose(imgs, (0,1,3,2,4))
        imgs = np.absolute(imgs).astype(np.float32)
        avg_diffs = np.zeros((imgs.shape[0]))
        for z in range(imgs.shape[0]):
            timeseries = imgs[z,:,:,:,:]
            diffs = np.zeros((timeseries.shape[0]-1, timeseries.shape[1], timeseries.shape[2], timeseries.shape[3]))
            for t in range(timeseries.shape[0]-1):
                diffs[t,:,:,:] = np.absolute(timeseries[t+1,:,:,:] - timeseries[t,:,:,:])
            avg_diffs[z] = np.mean(diffs)
        z_pruned = np.argsort(-avg_diffs) # sort in descending order
        max_imgs = np.amax(imgs, axis=(1,2,3))
        imgs = imgs/max_imgs[:,np.newaxis,np.newaxis,np.newaxis,:]
        for z in z_pruned[:20]:
            img = imgs[z,:,:,:,:]
            example = tf.train.Example(features=tf.train.Features(feature={
                'img': _bytes_feature(img.tostring())
            }))
            writer.write(example.SerializeToString())
            num_ex += 1
        print('patient {}'.format(i))
        print('example {}'.format(num_ex))
    writer.close()
    return

def what_tf_g7_adaproon(mode, patients):
    if os.path.exists('datasets/{}_g7_ada.tfrecords'.format(mode)):
        print('{}_g7_ada.tfrecords already exists'.format(mode))
        return
    num_ex = 0
    path_h5 = 'datasets/{}.h5'.format(mode)
    f = h5.File(path_h5, 'r')
    path_tf = 'datasets/{}_g7_ada.tfrecords'.format(mode)
    writer = tf.python_io.TFRecordWriter(path_tf)
    for i,p in enumerate(patients):
        imgs = f[p]
        imgs = np.transpose(imgs, (0,1,3,2,4))
        imgs = np.absolute(imgs).astype(np.float32)
        avg_diffs = np.zeros((imgs.shape[0]))
        for z in range(imgs.shape[0]):
            timeseries = imgs[z,:,:,:,:]
            diffs = np.zeros((timeseries.shape[0]-1, timeseries.shape[1], timeseries.shape[2], timeseries.shape[3]))
            for t in range(timeseries.shape[0]-1):
                diffs[t,:,:,:] = np.absolute(timeseries[t+1,:,:,:] - timeseries[t,:,:,:])
            avg_diffs[z] = np.mean(diffs)
        z_pruned = np.argsort(-avg_diffs) # sort in descending order
        max_imgs = np.amax(imgs, axis=(1,2,3))
        imgs = imgs/max_imgs[:,np.newaxis,np.newaxis,np.newaxis,:]
        thresh = int(round(0.25*imgs.shape[0]))
        for z in z_pruned[:thresh]:
            img = imgs[z,:,:,:,:]
            example = tf.train.Example(features=tf.train.Features(feature={
                'img': _bytes_feature(img.tostring())
            }))
            writer.write(example.SerializeToString())
            num_ex += 1
        print('patient {}'.format(i))
        print('example {}'.format(num_ex))
    writer.close()
    return

def create_dataset_g7(mode, patients):
    filenames = 'datasets/{}_g7.tfrecords'.format(mode)
    dataset = tf.data.TFRecordDataset(filenames)
    def _prep_tfrecord(example):
        features = tf.parse_single_example(
            example,
            features={'img': tf.FixedLenFeature([], tf.string)}
        )
        img_record_bytes = tf.decode_raw(features['img'], tf.float32)
        img = tf.reshape(img_record_bytes, [18, 192, 224, 1])
        features = img[:3,:,:,:]
        labels = img[3:,:,:,:]
        return features, labels
    dataset = dataset.map(_prep_tfrecord)
    return dataset

def create_dataset_g7_n(mode, patients, n):
    filenames = 'datasets/{}_g7.tfrecords'.format(mode)
    dataset = tf.data.TFRecordDataset(filenames)
    def _prep_tfrecord(example):
        features = tf.parse_single_example(
            example,
            features={'img': tf.FixedLenFeature([], tf.string)}
        )
        img_record_bytes = tf.decode_raw(features['img'], tf.float32)
        img = tf.reshape(img_record_bytes, [18, 192, 224, 1])
        features = img[:n,:,:,:]
        labels = img[n:,:,:,:]
        return features, labels
    dataset = dataset.map(_prep_tfrecord)
    return dataset

def create_dataset_g7_adaproon(mode, patients, n):
    filenames = 'datasets/{}_g7_ada.tfrecords'.format(mode)
    dataset = tf.data.TFRecordDataset(filenames)
    def _prep_tfrecord(example):
        features = tf.parse_single_example(
            example,
            features={'img': tf.FixedLenFeature([], tf.string)}
        )
        img_record_bytes = tf.decode_raw(features['img'], tf.float32)
        img = tf.reshape(img_record_bytes, [18, 192, 224, 1])
        features = img[:n,:,:,:]
        labels = img[n:,:,:,:]
        return features, labels
    dataset = dataset.map(_prep_tfrecord)
    return dataset

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #