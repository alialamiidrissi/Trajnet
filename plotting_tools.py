import numpy as np
import pandas as pd
from trajectories_trans_tools import *
import matplotlib
import matplotlib.pyplot as plt
import torch
import random
import os


def draw_heatmap(trajectories, size=32, min_bounds=None, max_bounds=None):
	'''
	Compute a  heatmap of the trajectories passed in parameter

	Parameters
	----------
	trajectories : pytorch tensor of size (nb_trajectories*nb_frames*2)
	size: dimension of the returned heatmap
	min_bounds : if not None, Tensor of size (2) specifying the minimum x and y positions. Else if it is None, this bounds are directly inferred from the given trajectories
	max_bounds : if not None, Tensor of size (2) specifying the maximum x and y positions. Else if it is None, this bounds are directly inferred from the given trajectories
	Returns
	-------
	grid : the Heatmap as as a torch tensor of size (size*size)
	min_bounds : Pytorch Tensor of size (2) specifying the minimum x and y positions
	max_bounds : Pytorch ensor of size (2) specifying the maximum x and y positions.
	increments : Pytorch Tensor of size (2) specifying the discretization steps for the x and y axis respectively
	'''
	grid = en_cuda(torch.zeros(size, size))
	reshaped_traj_tr = trajectories[:, :8, :][:, :, [2, 3]].view(-1, 2)
	reshaped_traj_te = trajectories[:, 8:, :][:, :, [2, 3]].view(-1, 2)
	reshaped_traj = trajectories[:, :, :][:, :, [2, 3]].view(-1, 2)
	if min_bounds is None:
		min_bounds, _ = torch.min(reshaped_traj, 0)
	if max_bounds is None:
		max_bounds, _ = torch.max(reshaped_traj, 0)
	increments = (max_bounds.sub(min_bounds)).div(size - 1)
	coordinates = torch.clamp(size - torch.floor(reshaped_traj_tr.sub(min_bounds).div(increments)).type(
	    torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor) - 1, 0, size - 1)
	print(torch.min(reshaped_traj, 0), torch.max(reshaped_traj, 0))
	indices = torch.mul(coordinates[:, 0], size).add(coordinates[:, 1])
	grid.put_(indices, en_cuda(torch.Tensor([1])).expand_as(indices))
	coordinates = size - torch.floor(reshaped_traj_te.sub(min_bounds).div(increments)).type(
	    torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor) - 1
	indices = torch.mul(coordinates[:, 0], size).add(coordinates[:, 1])
	grid.put_(indices, en_cuda(torch.Tensor(
	    [3])).expand_as(indices), accumulate=True)
	return grid.transpose(1, 0), (min_bounds, max_bounds, increments)


def draw_discretized_trajs(trajectories, size=32, min_bounds=None, max_bounds=None):
	'''
	compute a discretized representation of  trajectories passed in parameter

	Parameters
	----------
	trajectories : pytorch tensor of size (nb_trajectories*nb_frames*2)
	size: dimension of the returned heatmap
	min_bounds : if not None, Tensor of size (2) specifying the minimum x and y positions. Else if it is None, this bounds are directly inferred from the given trajectories
	max_bounds : if not None, Tensor of size (2) specifying the maximum x and y positions. Else if it is None, this bounds are directly inferred from the given trajectories
	Returns
	-------
	grid : The discretized space as a torch tensor of size (size*size)
	min_bounds : Pytorch Tensor of size (2) specifying the minimum x and y positions
	max_bounds : Pytorch ensor of size (2) specifying the maximum x and y positions.
	increments : Pytorch Tensor of size (2) specifying the discretization steps for the x and y axis respectively
	'''
	grid = en_cuda(torch.zeros(size, size))
	reshaped_traj_tr = trajectories[:, :8, :][:, :, [2, 3]].view(-1, 2)
	reshaped_traj_te = trajectories[:, 8:, :][:, :, [2, 3]].view(-1, 2)
	reshaped_traj = trajectories[:, :, :][:, :, [2, 3]].view(-1, 2)
	if min_bounds is None:
		min_bounds, _ = torch.min(reshaped_traj, 0)
	if max_bounds is None:
		max_bounds, _ = torch.max(reshaped_traj, 0)
	increments = (max_bounds.sub(min_bounds)).div(size - 1)
	coordinates = torch.clamp(size - torch.floor(reshaped_traj_tr.sub(min_bounds).div(increments)).type(
	    torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor) - 1, 0, size - 1)
	print(torch.min(reshaped_traj, 0), torch.max(reshaped_traj, 0))
	indices = torch.mul(coordinates[:, 0], size).add(coordinates[:, 1])

	grid.put_(indices, en_cuda(torch.Tensor([1])).expand_as(indices))
	coordinates = size - torch.floor(reshaped_traj_te.sub(min_bounds).div(increments)).type(
	    torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor) - 1
	indices = torch.mul(coordinates[:, 0], size).add(coordinates[:, 1])

	grid.put_(indices, en_cuda(torch.Tensor(
	    [3])).expand_as(indices), accumulate=True)
	return grid.transpose(1, 0), (min_bounds, max_bounds, increments)


def draw_all_trajectories(arr, xticks, yticks):
	'''
	Draw all the trajectories in arr

	Parameters
	----------
	arr : list of pytorch tensor of size (nb_frames,2)
	xticks: the xticks for the plot
	yticks : The yticks for the plot
	Returns
	-------

	'''
	plt.figure()
	idx = 0
	plot_label = True
	for elem in arr:
		idx += 1
		print(idx)
		elem = elem.cpu().numpy()
		if plot_label:
			plt.plot(elem[:, 2][:8], elem[:, 3][:8], 'r.-', label='Observed steps')
			plt.plot(elem[:, 2][7:], elem[:, 3][7:],
			         'g.-', label='Steps to be Predicted')
			plot_label = False
		else:
			plt.plot(elem[:, 2][:8], elem[:, 3][:8], 'r.-', label='Observed steps')
			plt.plot(elem[:, 2][7:], elem[:, 3][7:],
			         'g.-', label='Steps to be Predicted')
		if len(elem[:, 2]) != 20:
			print(elem.shape)
	plt.xticks(xticks)
	plt.yticks(yticks)
	plt.legend()


def plot_hists(folder_name, features, df, range_=None):
	'''
	Draw Histograms of the specified feature based on csv file generated by 'csv_specs_generators.py'

	Parameters
	----------
	folder_name : folder where the histograms will be saved
	features: The dataframe columns to use for the histogram
	df : Dataframe containg the specs for every trajectory
	range_ : The histograms bins

	Returns
	-------

	Example:

	plot_hists('distributions_mean_deviations',['Mean_rotation'],df,range_=range_) will generate histograms based on the feature 'Mean_rotation' when 'df' data is groupped by file
	and by dataset

	'''

	# Group trajectories specs by file
	for file, x in df.groupby('File'):
		fig, ax = plt.subplots()
		fig.set_size_inches(10, 10)
		data = []
		for feature in features:
			el = x[feature]
			if len(x[feature]) == 1:
				el = el.append(x[feature])
			data.append(el)

		if range_ is not None:
			plt.xticks(range_)
			ax.hist(data, bins=range_, label=features, alpha=0.5)
		else:
			ax.hist(el, label=features, ax=ax)

		_, labels = plt.xticks()
		plt.setp(labels, rotation=30)
		plt.legend()
		plt.savefig(folder_name + '/' +
		            '{}_{}.png'.format(x['Dataset'].iloc[0], file))
		plt.close()

	# Group trajectories specs by dataset
	for dts, x in df.groupby('Dataset'):
		fig, ax = plt.subplots()
		fig.set_size_inches(10, 10)
		data = []
		for feature in features:
			el = x[feature]
			if len(x[feature]) == 1:
				el = el.append(x[feature])
			data.append(el)

		if range_ is not None:
			plt.xticks(range_)
			ax.hist(data, bins=range_, label=features, alpha=0.5)
		else:
			ax.hist(el, label=features, ax=ax)
		plt.legend()
		_, labels = plt.xticks()

		plt.setp(labels, rotation=30)

		plt.savefig(folder_name + '/' + '{}.png'.format(x['Dataset'].iloc[0]))
		plt.close()


def plot_tracklet(tracklet, xlim=[-100, 100], ylim=[-100, 100], n=5):
	'''
	Plots a trajectory with its nearest n neighbors

	Parameters
	----------
	tracklet : tupple (trajectory,list of neighbors)
	size: dimension of the returned heatmap
	xlim : x axis limits
	ylim : y axis limits
	Returns
	-------

	'''
    fig, axes = plt.subplots()
    g_0 = None
    top_n = []
    sorting_list = []
    for neighbors in tracklet[1]:
        min_frame = neighbors[0,0]
        max_frame = neighbors[-1,0]
        center_track = tracklet[0][((tracklet[0][:,0]>=min_frame) & (tracklet[0][:,0]<=max_frame)).nonzero(),:]
        dist = torch.mean(torch.sum((center_track-neighbors)**2,1))
        sorting_list.append(dist)
    
    sorted_args = np.argsort(np.array(sorting_list))[:n]
    
    
    for neighb_idx in sorted_args:
        neighb = tracklet[1][neighb_idx].numpy()
        if g_0 is None:
            g_0, =axes.plot(neighb[:,2][:8],neighb[:,3][:8],'.-g',label = "Neigboring trajectories-observation ")
            axes.plot(neighb[:,2][7:],neighb[:,3][7:],'.-r',label = "Neigboring trajectories-prediction ")
        else:
            g_0, =axes.plot(neighb[:,2][:8],neighb[:,3][:8],'.-g')
            axes.plot(neighb[:,2][7:],neighb[:,3][7:],'.-r')
        
        if((neighb[0,2] >xlim[0]) and (neighb[0,2] <xlim[1]) and (neighb[0,3]>ylim[0]) and (neighb[0,3]< ylim[1])):
            axes.text(neighb[0,2],neighb[0,3],'S')
        if((neighb[-1,2] >xlim[0]) and (neighb[-1,2] < xlim[1]) and (neighb[-1,3]>ylim[0]) and (neighb[-1,3]< ylim[1])):
            axes.text(neighb[-1,2],neighb[-1,3],'E')

    rotated_traj = tracklet[0].numpy()
    axes.plot(rotated_traj[:,2][:8],rotated_traj[:,3][:8],'.-b',label = 'Center trajectory-observation')
    axes.plot(rotated_traj[:,2][7:],rotated_traj[:,3][7:],'.-m',label = 'Center trajectory-prediction')
    axes.text(rotated_traj[0,2],rotated_traj[0,3],'S')
    axes.text(rotated_traj[19,2],rotated_traj[19,3],'E')

    fig.set_figheight(7)
    fig.set_figwidth(7)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
