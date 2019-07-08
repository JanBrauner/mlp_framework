# Script Generator - generates shell scripts to be run as independant slurm jobs
from shutil import copyfile
import os
import json
import itertools
import numpy as np

#%% Run this cell once at the beginning

### default settings
default_args = {

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
"gpu_id": "0", # "A string indicating the gpu to use, ids separated by ','. For e.g. 4 gpus, this would usually be [0,1,2,3]."

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

default_script = """#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition={0}
#SBATCH --gres=gpu:{1}
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time={2}

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${{CUDNN_HOME}}/lib64:${{CUDA_HOME}}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${{CUDNN_HOME}}/lib64:$LIBRARY_PATH

export CPATH=${{CUDNN_HOME}}/include:$CPATH

export PATH=${{CUDA_HOME}}/bin:${{PATH}}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${{STUDENT_ID}}


export TMPDIR=/disk/scratch/${{STUDENT_ID}}/
export TMP=/disk/scratch/${{STUDENT_ID}}/

mkdir -p ${{TMP}}/datasets/
export DATASET_DIR=${{TMP}}/datasets/
# Activate the relevant virtual environment:
source /home/${{STUDENT_ID}}/miniconda3/bin/activate mlp
cd ..

mkdir /disk/scratch/${{STUDENT_ID}}/data/
rsync -ua --progress /home/${{STUDENT_ID}}/mlp_framework/data/ /disk/scratch/${{STUDENT_ID}}/data/
export DATASET_DIR=/disk/scratch/${{STUDENT_ID}}/data/

python main.py --experiment_name {3}
"""

### Paths 
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "configs"))
shell_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "cluster_experiment_scripts"))

### functions
def create_config_file(experiment_path, args):
    with open('{}.json'.format(experiment_path), 'w') as f:
        json.dump(args, f, indent=1)

def create_shell_script(experiment_name, experiment_path, partition, args, time=None):
    global default_script
    assert partition in ["Interactive","Standard"]
    assert type(args["gpu_id"]) == str
    num_gpus = len(args["gpu_id"].split(","))
    if time == None:
        if partition == "Interactive":
            time = "0-02:00:00"
        elif partition == "Standard":
            time = "0-08:00:00"
    script_str = default_script.format(partition, num_gpus, time, experiment_name)
    with open("{}.sh".format(experiment_name), "w") as f:
        f.write(script_str)
        
    

# =============================================================================
# #%% one experiment 
# experiment_name = "test_script_generator"
# partition = "Interactive"
# args = default_args
# time = None
# keys_to_update = ["num_epochs", "loss"]
# values_to_update = [50, "L2"]
# 
# # update args
# for key,value in zip(keys_to_update, values_to_update):
#     assert key in args.keys(), "wrong parameters name"
#     args[key] = value
#     
# create_config_file(os.path.join(config_path, experiment_name), args)
# create_shell_script(experiment_name=experiment_name,
#                     experiment_path=os.path.join(shell_script_path, experiment_name), 
#                                      partition=partition, args=args, time=time)
# =============================================================================
#%% one parameter over a range
experiment_base_name = "test_script_generator_range_{}" # needs to include {}
key_to_vary = "num_epochs"
values = [10,30,70]
partition = "Interactive"
args = default_args
time = None

# other non-default parameters
keys_to_update = ["num_epochs", "loss"]
values_to_update = [50, "L2"]

# update other non-default parameters
for key,value in zip(keys_to_update, values_to_update):
    assert key in args.keys(), "wrong parameters name"
    args[key] = value

# iterate over parameter range:
for value in values:
    experiment_name = experiment_base_name.format(value)
    args[key_to_vary] = value
    create_config_file(os.path.join(config_path, experiment_name), args)
    create_shell_script(experiment_name=experiment_name,
                    experiment_path=os.path.join(shell_script_path, experiment_name), 
                                     partition=partition, args=args, time=time)
    
    

# =============================================================================
# #%% Two-parameter grid
#     # to implement
#     # maybe use this thing:
# """
# From Joe:
# 
# import json
# import itertools
# import numpy as np
# 
# models=['resnet', 'densenet']
# weightdecays = np.linspace(1e-5, 1e-1, num=20)
# 
# 
# for i, (model, weightdecay) in enumerate(itertools.product(models, weightdecays)):
#     conf = {'model':model, 'weightdecay':weightdecay}
#     with open('conf{}.json'.format(i), 'w') as f:
#         json.dump(conf, f)
# """
# =============================================================================
