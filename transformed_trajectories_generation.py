import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import torch
import random
import functools
import seaborn as sns
import os
import matplotlib.ticker as ticker
from trajectories_trans_tools import *

# 
path = 'PATH TO TRAINING DATASET'

idx = 0

# dictionary mapping the datasets folders to their index in all_files
map_dataset_idx= {}

# dictionary mapping the training files to their index in all_files

map_file_idx= {}
all_files = []

for dataset in os.listdir(path):
	sub_idx = 0
	if( not dataset.startswith('.') and dataset not in ['validation']):
		print(dataset)
		dts = []
		for file in os.listdir(path+'/'+dataset):
			if( not file.startswith('.')):
				print(file)
				# Load tracklets and transform them 
				res= generate_tracklets(path+'/'+dataset+'/'+file)
				res = transform_tracklets_trajectories(res)

				# Append the transformed trajectories of a file to dts
				dts.append(res)
				map_file_idx[file] = sub_idx
				sub_idx +=1
		# Append the list of all transformed trajectories of a dataset to all_files
		all_files.append(dts)
		map_dataset_idx[dataset] = idx 
		idx += 1

# Save preprocessed trajectories

mapping_file_for_dataset = [[] for i in range(len(map_dataset_idx))]
for dataset in os.listdir(path):
	files =[]
	if( not dataset.startswith('.') and dataset not in ['validation']):
		for file in os.listdir(path+'/'+dataset):
			if(not file.startswith('.')):
				files.append(file)
		mapping_file_for_dataset[map_dataset_idx[dataset]] = files
		
if not os.path.exists('preprocessed_data'):
	os.makedirs('preprocessed_data')
	
for dts,idx in map_dataset_idx.items():
	if not os.path.exists('preprocessed_data/{}'.format(dts)):
		os.makedirs('preprocessed_data/{}'.format(dts))
	for file in mapping_file_for_dataset[idx]:
		for ped in all_files[idx][map_file_idx[file]]:
			np.savetxt('preprocessed_data/{}/{}_{}_neighbors.csv'.format(dts,file,int(ped[0][0,1])),torch.cat(ped[1],0).numpy() if len(ped[1]) else np.array([]),delimiter = ',')
			np.savetxt('preprocessed_data/{}/{}_{}_traj.csv'.format(dts,file,int(ped[0][0,1])),ped[0].numpy(),delimiter = ',')