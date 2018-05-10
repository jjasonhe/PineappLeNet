import os
import ast
import numpy as np
import cfl
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
		for f in folders:
			img = cfl.read('datasets/%s/im_dce' % f)
			img = np.squeeze(img)
			txt.write("%s," % (img.shape,))
		txt.close()
	txt = open("shapes.txt", "r")
	shp = txt.read()
	txt.close()
	shp = list(ast.literal_eval(shp))
	#print(shp)
	maxX = max(shp, key=itemgetter(4))[4]
	maxY = max(shp, key=itemgetter(3))[3]
	maxT = max(shp, key=itemgetter(0))[0]
	return maxX,maxY,maxT

def create_dicts():
	train_dict = list(folders[0:70])
	val_dict = list(folders[70:80])
	test_dict = list(folders[80:100])
	return train_dict,val_dict,test_dict

def fetch(f, maxX, maxY, maxT):
	img = cfl.read('datasets/%s/im_dce' % f)
	img = np.squeeze(img)
	if img.shape[4] < maxX:
		img = np.pad(img, ((0,),(0,),(0,),(0,),((maxX-img.shape[4])//2,)), mode='constant')
	if img.shape[3] < maxY:
		img = np.pad(img, ((0,),(0,),(0,),((maxY-img.shape[3])//2,),(0,)), mode='constant')
	img = img[:,0,:,:,:]
	img = np.transpose(img, (1, 3, 2, 0))
	img = img[:,::-1,::-1,:]
	return img