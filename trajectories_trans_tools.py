import numpy as np
import pandas as pd
import torch
import random
import functools
import os


def compute_speeds(a):
    '''
    Compute velocity vectors for a tensor of trajectories

    Parameters
    ----------
    a : pytorch tensor of size (nb_trajectories*frames*coordinates)

    Returns
    -------
    ret :pytorch tensor of size (nb_trajectories*frames*coordinates) with velocities, the first velocity of every trajectory
    is (0,0)
    '''
    ret = a[:, 1:, :] - a[:, :-1, :]
    ret = torch.cat(
        [en_cuda(torch.zeros(a.size()[0], 1, a.size()[2])), ret], 1)
    return ret


def create_rotation_matrices(angles, real=False):
    '''
    Create a rotation matrix with the corresponding angles passed in parameters

    Parameters
    ----------
    angles : torch vector with angles or scalar if real is True
    real : if True, this function considers that angles is a scalar and return only one rotation matrix 

    Returns
    -------
    ret : One or multiple rotation matrices
    '''
    # Rotate
    if real:
        angle = ((np.pi / 2) - angles)
        return en_cuda(torch.Tensor([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))
    matrices = []
    for i in range(angles.size()[0]):
        angle = ((np.pi / 2) - angles[i])
        matrices.append([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return en_cuda(torch.Tensor(matrices))


def rotate(rotation, points, init=False):
    '''
    Create a rotation matrix with the corresponding angles passed in parameters

    Parameters
    ----------
    rotations : torch Tensor of size (2*2) representing a rotation matrix
    points : torch tensor of size (nb_trajectories*nb_frames*2) if init is False else torch tensor of size (nb_frames*2)
    init : Boolean that controls the dimensions of points (see points description)
    Returns
    -------
    rotated_pts : rotated vectors using the given rotation matrix. The return object is a torch Tensor of size similar to 'rotations'
    '''
    rotated_pts = en_cuda(torch.zeros_like(points))
    if init:
        for i in range(points.size()[0]):
            rotated_pts[i, :] = torch.matmul(rotation, points[i, :])
        return rotated_pts.unsqueeze(0)

    for i in range(points.size()[0]):
        for j in range(points.size()[1]):
            rotated_pts[i, j, :] = torch.matmul(rotation, points[i, j, :])
    return rotated_pts


def en_cuda(tensor):
    '''
    Check if CUDA is available and modify a tensor accordingly

    Parameters
    ----------
    tensor : torch Tensor 

    Returns
    -------
    tensor : same tensor but with cuda enabled if CUDA is available
    '''
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def generate_tracklets(scene, factor=10, compute_neighbors=True, neighboring_dist=6, compute_specs=True):
    '''
    Group trajectories with their respective neighbors

    Parameters
    ----------
    scene : path for the text file containing the trajectories
    factor : factor used to scale the trajectory coordinates  
    compute_neighbors : If true the function looks for the neighbors of every trajectory and computes some related statistics
    neighboring_dist :  Max distance between a trajectory and its critical neighbors at a frame (used to compute number of critical neighbors) 
    Returns
    -------
    ret : list of tupples (trajectory,neighbors,spec)
    trajectory :the center trajectory
    neighbors : contain the 'trajectory' neighbors positions for every frame
    spec : dictionnary containing 
            - The neighbors IDs with key 'neighb'
            - The number of critical neighbors with key 'nb_critical_neighb'
            - The mean number of neighbors per frame 'mean_nb_neighbors_per_frame'
            - The mean number of critical neighbors per frame 'mean_nb_critical_neighbors_per_frame'
            - The mapping between ped_id in the file and its index in the returned list
    '''
    # Read data
    df_positions = pd.read_table(scene, delim_whitespace=True, header=None)
    df_positions.loc[:, [2, 3]] = df_positions.loc[:, [2, 3]] * factor
    frame_pos_mappings = {}
    grp_by_frames = []
    neighboring_dist = neighboring_dist * factor
    neighboring_dist = neighboring_dist**2
    # Group by frame, store groups in "grp_by_frames" and map frame number to
    # location in "grp_by_frames" using "unique_ped_in_frame"
    unique_ped_in_frame = {}
    idx = 0
    for frame, x in df_positions.groupby(df_positions[0]):
        grp_by_frames.append(x)
        frame_pos_mappings[frame] = idx
        idx += 1
        unique_ped_in_frame[frame] = x[1]

    # Group by pedestrian
    ped_id_grp_pos_mapping = {}
    grp_by_ped_id = []
    idx = 0
    for ped_id, x in df_positions.groupby(df_positions[1]):
        grp_by_ped_id.append(x)
        ped_id_grp_pos_mapping[ped_id] = idx
        idx += 1

    final_tracklets = []
    # Iterate over all pedestrians positions and find their neighbors if
    # compute_neighbors is True
    for x in grp_by_ped_id:
        set_neighb = set()
        min_frame = x.iloc[0, 0]
        max_frame = x.iloc[19, 0]
        buffer_neighbors = []
        mean_nb_neighbors = 0
        mean_nb_critical_neighbors = [0] * 20
        if compute_neighbors:
            # Find x neighbors
            for frame_id in x[0]:
                neighb = grp_by_frames[
                    frame_pos_mappings[frame_id]][1].values.tolist()
                mean_nb_neighbors += len(neighb)
                set_neighb.update(neighb)
            nb_critical_neighbors = 0
            mean_contact = 0
            set_neighb.remove(x.iloc[0, 1])
            for neighb_id in set_neighb:
                neighb = grp_by_ped_id[ped_id_grp_pos_mapping[neighb_id]]
                select = neighb[(neighb[0] >= min_frame) & (
                    neighb[0] <= max_frame)].as_matrix()
                # distance of neighbors to center pedestrian
                if compute_specs:
                    idx_frame = None
                    is_critical_neighb = False
                    for neighb_pos in select:
                        if idx_frame is None:
                            idx_frame = (x[0] == neighb_pos[0]).nonzero()[0][0]
                        else:
                            idx_frame += 1
                        distance = (neighb_pos[2] - x.iloc[idx_frame, 2]
                                    )**2 + (neighb_pos[3] - x.iloc[idx_frame, 3])**2
                        # Check if the neighbor is close to the center
                        # trajectory
                        if distance <= neighboring_dist:
                            mean_contact += 1
                            mean_nb_critical_neighbors[idx_frame] += 1
                            is_critical_neighb = True
                    if is_critical_neighb:
                        nb_critical_neighbors += 1
                buffer_neighbors.append(en_cuda(torch.Tensor(select)))

            spec = {}
            if compute_specs:
                spec['neighb'] = set_neighb
                spec['nb_critical_neighb'] = nb_critical_neighbors
                spec['mean_nb_neighbors_per_frame'] = mean_nb_neighbors / 20
                spec['mean_nb_critical_neighbors_per_frame'] = functools.reduce(
                    lambda x, y: x + y, mean_nb_critical_neighbors) / 20.0
                spec['mean_contact'] = mean_contact / \
                    float(nb_critical_neighbors) if nb_critical_neighbors > 0 else 0
                spec['ped_id_grp_pos_mapping'] = ped_id_grp_pos_mapping
            final_tracklets.append(
                (en_cuda(torch.Tensor(x.as_matrix())), buffer_neighbors, spec))
        else:
            final_tracklets.append(
                (en_cuda(torch.Tensor(x.as_matrix())), buffer_neighbors, {}))
    return final_tracklets


def transform_tracklets_trajectories(trajs, compute_neighbors=True):
    '''
    Transform each tracklet (trajectory and its neighbors) by rotating the center trajectory and its neighbors 
    around the center trajectory first point so that the center trajectory starts facing up. Then translate the tracklet trajectories
    so that the center trajectory start at point (0,0)

    Parameters
    ----------
    trajs : tracklets list of tupples (trajectory,neighbors,spec). See documentation for generate_tracklets function in order to have more informations
    compute_neighbors : if True, apply transformations on trajectory neighbors
    Returns
    -------
    ret : list of tupples (trajectory,neighbors,spec)
    trajectory :the transformed center trajectory
    neighbors : contain the transformed 'trajectory' neighbors positions for every frame if compute_neighbors is True
    spec : dictionnary containing 
            - The neighbors IDs with key 'neighb'
            - The number of critical neighbors with key 'nb_critical_neighb'
            - The mean number of neighbors per frame 'mean_nb_neighbors_per_frame'
            - The mean number of critical neighbors per frame 'mean_nb_critical_neighbors_per_frame'
            - The mapping between ped_id in the file and its index in the returned list
    '''
    transformed_traj = []
    for traj in trajs:
        center_point = traj[0][0][[2, 3]]
        angle = np.arctan2(traj[0][1, 3] - center_point[1],
                           traj[0][1, 2] - center_point[0])
        rotation_matrix = create_rotation_matrices(angle, real=True)
        rotated_pos_c = rotate(rotation_matrix, traj[0][:, [2, 3]].sub(
            center_point).unsqueeze(0)).squeeze(0)
        rotated_pos_c = torch.cat([traj[0][:, [0, 1]], rotated_pos_c], 1)
        buffer_neighb = []
        if compute_neighbors:
            for neighb in traj[1]:
                rotated_pos = rotate(rotation_matrix, neighb[:, [2, 3]].sub(
                    center_point).unsqueeze(0)).squeeze(0)
                rotated_pos = torch.cat([neighb[:, [0, 1]], rotated_pos], 1)
                buffer_neighb.append(rotated_pos)
        if compute_neighbors:
            transformed_traj.append((rotated_pos_c, buffer_neighb, traj[2]))
        else:
            transformed_traj.append((rotated_pos_c, traj[1], traj[2]))
    return transformed_traj
