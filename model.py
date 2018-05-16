import numpy as np
import tensorflow as tf

def construct_model(curr,
					iter_num=-1.0,
					k=-1,
					num_masks=10,
					stp=False,
					cdna=True,
					dna=False,
					context_frames=1):
	'''
	Args:
		curr: tensor of ground truth "current time" images
              in my case, h5file[dictionary[i]], which is of shape (Z*T, Y, X)
		iter_num: tensor of the current training interation
		k: constant used for scheduled sampling, -1 to feed in own prediction
		num_masks: the number of different pixel motion predictions
		stp: True to use Spatial Transformer Predictor (STP)
		cdna: True to use Convolutional Dynamic Neural Advection (CDNA)
		dna: True to use Dynamic Neural Advection (DNA)
		context_frames: number of ground truth frames to pass in before
						feeding predictions
	
	Returns:
		next_gen: predicted future image frames

	Raises:
		ValueError: if more than one network option specified or
					more than 1 mask specified for DNA model
	'''
	if stp+cdna+dna != 1:
		raise ValueError('More than one, or no network option specified.')
	N, H, W= curr.shape
    
    lstm_size = np.int32(np.array([32, 32, 64, 64, 128, 64, 32]))
    
    for img in curr:
        


def construct_model()
