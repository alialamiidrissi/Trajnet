# Trajnet tools and preprocessed datasets

This repository contains functions to preprocess and plot the datasets for the Trajnet challenge

## How to generate the preprocessed data and the 'dataset_specs.csv' file ?

- To generate the preprocessed data, run the 'transformed\_trajectories\_generation.py' script as follow :

`python transformed_trajectories_generation.py --output_folder [output_folder_path] --path_to_data [path_to_the_original_Trajnet_datasets]`

- To generate the'dataset_specs.csv' file, run the 'csv\_specs\_generator.py' script as follow :

`python csv_specs_generator.py --output_file [output_csv_file_name] --path_to_data [path_to_the_original_Trajnet_datasets]`


## Files description

- **plotting\_tools.py**: Python functions to plots the datasets
- **trajectories\_trans\_tools.py**: Python functions to preprocess the datasets
- **transformed\_trajectories\_generation.py**: Python script to generate the preprocessed datasets
- **csv\_specs\_generator.py**: Python script to generate the 'stat\_dataset' csv file
- **Preprocessed\_data.zip**: Zip file containing the preprocessed datasets and has the following structure:
	- Every folder contains all trajectories for a given dataset
	- Every preprocessed trajectory is saved in a separate file named '[file]\_[ped\_id]\_traj.csv'
	- All the neighbors of a specific trajectory are saved in a file named '[file]\_[ped\_id]\_neighbors.csv'
- **dataset\_specs.csv**: CSV file containing information about every pedestrian in every dataset.Its columns are described below:
	- **Dataset**: This column specifies the pedestrian dataset
	- **File**: This column specifies the pedestrian file
	- **Track\_ID**: This column specifies the pedestrian ID
	- **Mean\_rotation**: This column specifies the mean rotation of the pedesterian trajectory with respect to the vector between its first and second position
	- **nb\_neighbors** : This column specifies the current pedestrian number of neighbors 
	- **nb\_critical\_neighbors** : This column specifies the number of neighbors that are at most 6 meters far from the current pedestrian
	- **mean\_nb\_neighbors\_per\_frame**: This column specifies the mean number of neighbors of the current pedestrian per frame 
	- **mean\_nb\_critical\_neighbors\_per\_frame**: This column specifies the mean number of neighbors per frame that are at most 6 meters far from the current pedestrian
	- **mean\_contact** : This columns specifies the mean number of frames during which the current pedestrians appears with a neighbor
	- **is\_static** : This column specifies if the current pedestrian does not move
