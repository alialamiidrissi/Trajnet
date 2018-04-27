import numpy as np 
import pandas as pd 
import torch
import random
import functools
import os
from trajectories_trans_tools import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--path_to_data", help="Path to the original Trajnet data")
parser.add_argument("--output_folder", help="output transfomed data folder")
args = parser.parse_args()
path = args.path_to_data


output_folder = args.output_folder	
if not os.path.exists(output_folder):
	os.makedirs(output_folder)

for dataset in os.listdir(path):
	os.makedirs('{}/{}'.format(output_folder,dataset))
	if( not dataset.startswith('.') and dataset not in ['validation']):
		print('transforming {} dataset'.format(dataset))
		for file in os.listdir(path+'/'+dataset):
			if( not file.startswith('.')):
				print('\ttransforming {}'.format(file))
				# Load tracklets and transform them 
				res= generate_tracklets(path+'/'+dataset+'/'+file)
				res = transform_tracklets_trajectories(res)
				for ped in res:
					np.savetxt('{}/{}/{}_{}_neighbors.csv'.format(output_folder,dataset,file,int(ped[0][0,1])),torch.cat(ped[1],0).numpy() if len(ped[1]) else np.array([]),delimiter = ',')
					np.savetxt('{}/{}/{}_{}_traj.csv'.format(output_folder,dataset,file,int(ped[0][0,1])),ped[0].numpy(),delimiter = ',')

print('Transformed data saved')


