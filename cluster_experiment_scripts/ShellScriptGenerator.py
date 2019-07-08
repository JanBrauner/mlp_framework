# Script Generator - generates shell scripts to be run as independant slurm jobs
from shutil import copyfile
import os
import json
import itertools
import numpy as np

#%% Run this cell once at the beginning

### default settings
default = {

# =============================================================================
### experiment settings

"continue_from_epoch": -1, # "-1": start from scratch. -2": start from model with best validation set performance. Any positive integer": start from specified epoch"

# =============================================================================
### misc parameters
"seed": 0,
"task": "regression", #"Choose training task from "regression", "classification"

# =============================================================================
### GPU settings
"use_gpu": True, # 'A flag indicating whether we will use GPU acceleration or not'
"gpu_id": "0", # "A string indicating the gpu to use, ids separated by ','"

# =============================================================================
### model parameters
"model_name": "context_encoder",

# model parameters: convolutions
"kernel_size": 4,

# model parameters: encoder
"num_layers_enc": 6,
"num_channels_enc": 64,
"num_channels_progression_enc": [1,1,2,4,8],
"num_channels_bottleneck": 4000,

# model parameters: decoder
"num_layers_dec": 5,
"num_channels_dec": 64,
"num_channels_progression_dec": [8,4,2,1],

# =============================================================================
### data parameters
# data parameters: dataset
"dataset_name": "MiasHealthy",
"num_image_channels": 1,
"image_height": 128,# "Height of input images. If patches are used as input, this is the patch height, not the full image height."
"image_width": 128, #"Width of input images. If patches are used as input, this is the patch width, not the full image width."

# data parameters: misc
"debug_mode": False,
"num_workers": 4, 

# data parameters: augmentations
"augment": False,
"gamma_factor": 1,
"rot_angle": 0,
"translate_factor": [0, 0],
"scale_factor": 1,
"shear_angle": 0,

# data parameters: image patches
"patch_size": [128, 128],
"patch_location_during_training": "central",

# data parameters: masking
"mask_size": [64, 64],

# =============================================================================
### training parameters
# training parameters: general
"batch_size": 50,
"loss": "L2",
"num_epochs": 100,

# training parameters: optimiser
"learning_rate": 0.0002,
"betas": [0.5, 0.999],
"weight_decay_coefficient": 0

# =============================================================================

}


### Paths 
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "configs"))


### functions
def create_config_file(experiment_name, args):
    with open('{}.json'.format(experiment_name), 'w') as f:
        json.dump(args, f, indent=1)


#%% one experiment 
experiment_name = "test_script_generator"
args = default
keys_to_update = ["num_epochs", "loss"]
values_to_update = [50, "L2"]

for key,value in zip(keys_to_update, values_to_update):
    assert key in args.keys(), "wrong parameters name"
    args[key] = value
    
create_config_file(os.path.join(config_path, experiment_name), args)

#%%
# =============================================================================
# 
# default_config_path = os.path.join(config_path, "CE_default_cluster.json")
# 
# 
# with open(default_config_path, "r") as f:
#     default = json.load(f)
#     
# 
# =============================================================================
# =============================================================================
# seed = '0'
# 
# def makeFiles(scriptNames, loss_weights):
# 
# 	for counter, scriptName in enumerate(scriptNames):
# 
# 	    # Open template with main preamble
# 	    copyfile('gpu_cluster_template.sh', scriptName+".sh")
# 
# 	    # Open for editing
# 	    f = open(scriptName+".sh","a")
# 
# 	    print(scriptName)
# 
# 	    # Append key arguments to final line
# 	    f.write("--batch_size 50 --continue_from_epoch -1 --seed " + seed + " --image_num_channels 1 --image_height 256 --image_width 256 --dim_reduction_type 'max_pooling' --num_layers 5 --num_filters 64 --num_epochs 100 --experiment_name '" + scriptName + "_256_s1' --use_gpu 'True' --gpu_id '0,1,2,3,4,5' --weight_decay_coefficient 0.0001 --dataset_name 'rsna' --mode 'multitask with 2 classes' --is_dense_net 1 --growth_rate 12 --drop_rate 0.3 --block_config 6 12 16 12 --bn_size 2 --block_config_dec 4 6 8 6 --simple_dec 'True' --bn_decoder 64 --loss_weights " + str(loss_weights[counter]) + " --loss_function 'bce' --num_init_features 64 --use_augmentation 'True' --gamma_factor 1.5 --rot_angle 6 --shear_angle 6 --translate_distance 20 --scale_factor 1.2 --oversampling 'False'")
# 
# 	    f.close()
# 
# # Lists of variables to iterate over 
# scriptNames = ['r10b_simpledec_CE_0p01_s1', 'r10b_simpledec_CE_0p1_s1', 'r10b_simpledec_CE_0p25_s1', 'r10b_simpledec_CE_0p5_s1', 'r10b_simpledec_CE_0p75_s1', 'r10b_simpledec_CE_0p9_s1', 'r10b_simpledec_CE_0p99_s1', 'r10b_simpledec_CE_1_s1']
# 
# loss_weights = ['0.01 0.99', '0.1 0.9', '0.25 0.75', '0.5 0.5', '0.75 0.25', '0.9 0.1', '0.99 0.01', '1 0']
# 
# makeFiles(scriptNames, loss_weights)
# 
# =============================================================================

"""
ToDo:
Functionality:
    - have one default json file to load from
    - update parameters there-in
        - single
        - grid
    - multiple seeds -> replicates
    - num_GPUs shouhld work automatically
    
    
    
    
    create json 
"""