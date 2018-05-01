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
parser.add_argument("--output_file", help="output csv file path and name")
args = parser.parse_args()
path = args.path_to_data


def compute_relative_angle(trajectories):
    '''
    Compute relative orientation of the trajectories passed in parameter with respect to a vector facing up (having an angle of 90 degrees with the x-axis )

    Parameters
    ----------
    trajectories : pytorch tensor of size (nb_trajectories*nb_frames*2)

    Returns
    -------
    mean_angles : Mean rotation of the trajectory with respect to a vector facing up across all frames for every trajectory
    max_angles : Maximum rotation of the trajectory with respect to a vector facing up across all frames for every trajectory
    is_static : List of boolean values of size nb_trajectory which determines if a pedestrian does not move during the observed nb_frames
    '''
    speeds = compute_speeds(trajectories[:, :, [2, 3]])[:, 1:, :]
    # Remove static positions
    cond = (torch.sqrt(speeds[:, :, 0]**2 + speeds[:, :, 1]**2)) > 1
    trajectories_without_stops = []
    is_static = []
    for idx, traj in enumerate(speeds):
        is_static_t = False
        # Check if the pedestrian does not move
        if (len(cond[idx, :].nonzero().size()) == 0):
            trajectories_without_stops.append(en_cuda(torch.Tensor([[0, 1]])))
            is_static_t = True
        else:
            trajectories_without_stops.append(
                traj[cond[idx, :].nonzero().squeeze(1), :])
        is_static.append(is_static_t)

    mean_angles = []
    max_angles = []
    # Compute angle wrt starting position
    for idx in range(len(trajectories_without_stops)):
        angles = torch.abs(torch.atan2(trajectories_without_stops[idx][
                           :, 1], trajectories_without_stops[idx][:, 0]) - (0.5 * np.pi))
        angles[angles > np.pi] = 2 * np.pi - angles[angles > np.pi]
        mean_angles.append(torch.mean(angles))
        max_angles.append(torch.max(angles))

    return np.degrees(np.array(mean_angles)), np.degrees(np.array(max_angles)), is_static

# Dataframe to save
df_infos = []
for dataset in os.listdir(path):
    if(not dataset.startswith('.') and dataset not in ['validation']):
        print('Reading {} dataset'.format(dataset))
        dts = []
        for file in os.listdir(path + '/' + dataset):
            if(not file.startswith('.')):
                print('\tReading {}'.format(file))
                # Load tracklets and transform them
                res = generate_tracklets(path + '/' + dataset + '/' + file)
                rotated = []
                nb_critical_neigbors, nb_neigbors, mean_contact, mean_nb_neighbors, mean_nb_critical_neighbors = [], [], [], [], []
                # For every transformed tracklet append the corresponding infos
                # to the dataframe to Save (generate_tracklets doc for more
                # infos)
                for result in transform_tracklets_trajectories(res, compute_neighbors=False):
                    rotated.append(result[0].unsqueeze(0))
                    nb_critical_neigbors.append(
                        result[2]['nb_critical_neighb'])
                    nb_neigbors.append(len(result[2]['neighb']))
                    mean_nb_neighbors.append(
                        result[2]['mean_nb_neighbors_per_frame'])
                    mean_nb_critical_neighbors.append(
                        result[2]['mean_nb_critical_neighbors_per_frame'])
                    mean_contact.append(result[2]['mean_contact'])

                rotated = torch.cat(rotated, 0)

                # Compute the mean and max deviation of the trajectories with respect to the vector (0,1)
                # (Which has a similar orientation to the vector between the first point and second point of every trajectory)
                mean, max_, is_static = compute_relative_angle(rotated)

                # tracklet specs
                df_tmp = pd.DataFrame(columns=['Dataset', 'File', 'Track_ID', 'Mean_rotation', 'nb_neighbors', 'nb_critical_neighbors',
                                               'mean_nb_neighbors_per_frame', 'mean_nb_critical_neighbors_per_frame', 'mean_contact', 'is_static'])
                df_tmp['Mean_rotation'] = mean
                df_tmp['Dataset'] = dataset
                df_tmp['File'] = file
                df_tmp['Track_ID'] = rotated[:, 0, 1].cpu().numpy()
                df_tmp['nb_neighbors'] = nb_neigbors
                df_tmp['mean_contact'] = mean_contact
                df_tmp['nb_critical_neighbors'] = nb_critical_neigbors
                df_tmp['is_static'] = is_static
                df_tmp['mean_nb_neighbors_per_frame'] = mean_nb_neighbors
                df_tmp[
                    'mean_nb_critical_neighbors_per_frame'] = mean_nb_critical_neighbors
                df_infos.append(df_tmp)
# Save final Dataframe
df = pd.concat(df_infos).reset_index().drop(['index'], axis=1)
df.to_csv(args.output_file)
print('CSV file saved')
