# Script Generator - generates shell scripts to be run as independant slurm jobs
from shutil import copyfile
import os
import json
import itertools
import numpy as np
import copy

import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from arg_extractor import get_args
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
"gpu_id": "0,1,2,3", # "A string indicating the gpu to use, ids separated by ','. For e.g. 4 gpus, this would usually be [0,1,2,3]."

# =============================================================================
### model parameters
"model_name": "context_encoder", # current options: context_encoder: Note that the context encoder can also be used as an autoencoder, when "num_layers_enc" == "num_layers_dec", and the input images are not masked out.

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
"image_height": 128,# Height of input images. If patches are used as input, this is the patch height, not the full image height.
"image_width": 128, # Width of input images. If patches are used as input, this is the patch width, not the full image width.
"normalisation": "range-11", # "mn0sd1" normalises to mean=0, std=1. "range-11" normalises to range [-1,1] 
"scale_image": None, # Set None for no adaptive scaling, otherwise specify the tuple of scaling factors for the image dimensions. Allows for adaptive scaling of the images. E.g. (0.5,0.5) shrinks every image dimension to half of its original size (using Image.resize). Can be useful if input images have variable size and I don't want to bring them all to a fixed size, e.g. because patch mode is used). Was not intended to be used with InpaintingDataset but patch_mode=False, probably causes some funny bugs if used like that
"data_format": "inpainting", # Options: 
                            # "inpainting": input image/patch is masked out, and the target is the content of the masked image. 
                            # "autoencoding": input image/patch == output image, no masking.
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
"patch_mode": True, # if true, patches of patch_size will be extracted from the image for training. If false, the whole image will be resized (possibly warping height/width relations) to (image_height,image_width)
"patch_size": [128, 128],
"patch_location_during_training": "random", # Can be "central" or "random"
"patch_rejection_threshold": 10, # CURRENTLY NOT USED!.threshold, on a 8-bit scale. Patches sampled from the data loader with a mean below this threshold get rejected because they show only background
"image_padding_mode": None, # Options: "constant", "edge" (different modes of padding, see torchvision.transforms.Pad). None if no padding required. The amount of padding is infered automatically as padding = max((args.patch_size[0] - args.mask_size[0])//2, (args.patch_size[1] - args.mask_size[1])//2) #automatically infer padding from patch and mask size. Padding only really makes sense with patch_mode==True


# data parameters: masking
"mask_size": [64, 64],

# =============================================================================
### training parameters
# training parameters: general
"batch_size": 50,
"loss": "L2", #  currently implemented: "L2" for regression, "cross_entropy" for classification
"num_epochs": 200,

# training parameters: optimiser
"learning_rate": 0.0002,
"betas": [0.5, 0.999],
"weight_decay_coefficient": 0,

# =============================================================================
### Parameters related to anomaly detection process
# Experiment parameters
"anomaly_dataset_name" : "MiasPathological", # Name of the dataset with anomalies

# Anomaly detection parameters
"AD_patch_stride": [10,10], # stride of the sliding window in image dimensions 0 and 1.
"measure_of_anomaly": "absolute distance", # current options: "absolute distance" (for regression models), "likelihood"(for model trained on cliassification)
"window_aggregation_method": "mean", # How to aggregate the results from overlapping sliding windows. Current option: "mean", "min", "max"
"save_anomaly_maps": True, # whether to save the anomaly score heat maps
"AD_margins" : None, # Tupel of image margins in image dimensions 1 and 2 that should not be considered for calculating agreement between anomaly map and label image
# computational parameters
"AD_batch_size": 50 # batch size for anomaly detection: how many sliding windows to load at the same time

# =============================================================================
}


# These are the only args that will be set to the default value listed above. All other args will be copied over from the train experiment. 
# The reason is that these args don't interact with the train setting. But e.g. anomaly dataset interacts with the train setting (you should choose the dataset you trained on), measure_of_anomaly also does, ...
args_to_keep_from_AD_experiment = [
        # anomaly detection specific args
        "window_aggregation_method",
        "save_anomaly_maps",
        "AD_batch_size",
        # computational args:
        "seed",
        "use_gpu",
        "gpu_id",
        "debug_mode",
        "num_workers"] 




default_script = """#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time={time}

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


"""

script_block_to_transfer_dataset = """mkdir -p /disk/scratch/${{STUDENT_ID}}/data/{dataset_name}/
rsync -ua --progress --delete /home/${{STUDENT_ID}}/mlp_framework/data/{dataset_name}/ /disk/scratch/${{STUDENT_ID}}/data/{dataset_name}/
export DATASET_DIR=/disk/scratch/${{STUDENT_ID}}/data/

"""

script_block_to_transfer_anomaly_dataset = """mkdir -p /disk/scratch/${{STUDENT_ID}}/data/{anomaly_dataset_name}/
rsync -ua --progress --delete /home/${{STUDENT_ID}}/mlp_framework/data/{anomaly_dataset_name}/ /disk/scratch/${{STUDENT_ID}}/data/{anomaly_dataset_name}/
export DATASET_DIR=/disk/scratch/${{STUDENT_ID}}/data/

"""

script_block_to_execute_training = "python main.py --experiment_name {experiment_name}"
script_block_to_execute_AD = "python anomaly_detection.py --experiment_name {experiment_name}"

### functions
def create_config_file(experiment_path, args):
    with open('{}.json'.format(experiment_path), 'w') as f:
        json.dump(args, f, indent=1)

def create_shell_script(experiment_name, experiment_type, experiment_path, partition, args, time=None):
    global default_script
    assert partition in ["Interactive","Standard"]
    assert type(args["gpu_id"]) == str
    assert experiment_type in ["train", "AD", "train+AD"]
    num_gpus = len(args["gpu_id"].split(","))
    
    # use default times if time is not given
    if time == None:
        if partition == "Interactive":
            time = "0-02:00:00"
        elif partition == "Standard":
            time = "0-08:00:00"
    
    # fill arguments into default string
    script_str = default_script.format(partition=partition, num_gpus=num_gpus, time=time, dataset_name=args["dataset_name"])
    
    # Depending on experiment type, choose the correct data to copy to computational node, and the correct python file to execute    
    if experiment_type == "train":
        script_str = script_str + script_block_to_transfer_dataset.format(dataset_name=args["dataset_name"])
        script_str = script_str + script_block_to_execute_training.format(experiment_name=experiment_name)
    elif experiment_type == "AD":
        script_str = script_str + script_block_to_transfer_anomaly_dataset.format(anomaly_dataset_name=args["anomaly_dataset_name"])
        script_str = script_str + script_block_to_execute_AD.format(experiment_name=experiment_name)
    elif experiment_type == "train+AD":
        script_str = script_str + script_block_to_transfer_dataset.format(dataset_name=args["dataset_name"])
        script_str = script_str + script_block_to_transfer_anomaly_dataset.format(anomaly_dataset_name=args["anomaly_dataset_name"])
        script_str = script_str + script_block_to_execute_training.format(experiment_name=experiment_name) + "\n"
        script_str = script_str + script_block_to_execute_AD.format(experiment_name=experiment_name)
       
        
    with open("{}.sh".format(experiment_name), "w") as f:
        f.write(script_str)
        

### Paths 
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "configs"))
shell_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "cluster_experiment_scripts"))

### Commonly used themes
# Misc themes
def cpu_theme(args):
    args["use_gpu"] = False
    args["num_workers"] = 0
    args["debug_mode"] = True
    args["batch_size"] = 5
    args["num_epochs"] = 2
    args["AD_batch_size"] = 5
    args["AD_patch_stride"] = (30,30)
    return args

def AD_theme(args):
    args["gpu_id"] ="0,1,2"
    args["num_workers"] = 6
    return args

def probabilistic_inpainting_theme(args):
    args["task"] = "classification"
    args["loss"] = "cross_entropy"
    args["measure_of_anomaly"] = "likelihood"
    return args

def small_mask_theme(args):
    args["num_layers_dec"] = 4
    args["num_channels_progression_dec"] = [8,4,2]
    args["mask_size"] = (32,32)
    return args

def large_context_theme(args):
    args["image_height"] = 220
    args["image_width"] = 220
    args["patch_size"] = (220,220)
    args["batch_size"] = 25
    return args

### Themes regarding autoencoders (might not be combinable with other themes):
def autoencoder_theme(args): # autoencoder on 128x128 patches
    args["num_layers_dec"] = 6
    args["num_channels_progression_dec"] = [8,4,2,1,1]
    args["data_format"] = "autoencoding"
    return args

def autoencoder_small_theme(args): # autoencoder on 64x64 patches
    args["image_height"] = 64 
    args["image_width"] = 64 
    args["patch_size"] = [64,64]
    args["num_layers_enc"] = 5
    args["num_channels_progression_enc"] = [1,2,4,8]
    args["data_format"] = "autoencoding"
    return args

### Dataset themes
def Mias_theme(args):
    args["num_image_channels"] = 1
    args["dataset_name"] = "MiasHealthy"
    args["anomaly_dataset_name"] = "MiasPathological"
    args["gpu_id"] ="0,1,2,4,5,6"
    args["num_workers"] = 12
    args["scale_image"] = (0.5, 0.5)
    return args
    
def GoogleStreetView_theme(args):
    args["num_image_channels"] = 3
    args["dataset_name"] = "GoogleStreetView"
    args["gpu_id"] ="0,1,2,3,4,5"
    args["num_workers"] = 12
    return args


def DescribableTextures_theme(args):
    args["num_image_channels"] = 3
    args["dataset_name"] = "DescribableTextures"
    args["anomaly_dataset_name"] = "DTPathologicalIrreg1"
    args["gpu_id"] ="0,1,2,3,4,5"
    args["num_workers"] = 12
    args["AD_margins"] = [128,128]
    return args
    

#%% A list of independent experiment 
# experiment names
experiment_names = ["r4_CE_Mias_augtest_ctrl",
                    
                    "r4_CE_Mias_augtest_flip", 
                    
                    "r4_CE_Mias_augtest_gamma_1p05",
                    "r4_CE_Mias_augtest_gamma_1p2",
                    "r4_CE_Mias_augtest_gamma_1p5",
                    "r4_CE_Mias_augtest_gamma_2",
                    
                    "r4_CE_Mias_augtest_scale_1p05",
                    "r4_CE_Mias_augtest_scale_1p2",
                    "r4_CE_Mias_augtest_scale_1p5",
                    
                    "r4_CE_Mias_augtest_rot_6",
                    "r4_CE_Mias_augtest_rot_15",
                    "r4_CE_Mias_augtest_rot_30",
                    
                    "r4_CE_Mias_augtest_shear_6",
                    "r4_CE_Mias_augtest_shear_15",
                    "r4_CE_Mias_augtest_shear_30",
                    
                    "r4_CE_Mias_augtest_combo_MLP",]
# Note: For experiments that include anomaly detection, the experiment name needs to be original_experiment_name + "___" + AD_experiment_name, where original_experiment_name is the name of the eperiment in which the model that we want to use for AD was trained.

# type of experiment
experiment_type = "train" # options: "train" for training (including evaluation on val and test set); "AD" for anomaly detection (using the best validation model from "experiment_name"); "train+AD" for both.

# number of replicates
num_replicates = 3

# Commonly used themes
cpu = False
probabilistic_inpainting = True
small_mask = False
large_context = False
Mias = True
DescribableTextures = False
GoogleStreetView = False
autoencoder = False
autoencoder_small = False

# slurm options
partition = "Standard"
time = None



# arguments to update from default, each inner dict has the items for one experiment:

update_dicts = [{},
                
                {"augment":True},
                
                {"augment":True, "gamma_factor":1.05},
                {"augment":True, "gamma_factor":1.2},
                {"augment":True, "gamma_factor":1.5},
                {"augment":True, "gamma_factor":2},
                
                {"augment":True, "scale_factor":1.05},
                {"augment":True, "scale_factor":1.2},
                {"augment":True, "scale_factor":1.5},
                
                {"augment":True, "rot_angle": 6},
                {"augment":True, "rot_angle": 15},
                {"augment":True, "rot_angle": 30},
                
                {"augment":True, "shear_angle": 6},
                {"augment":True, "shear_angle": 15},
                {"augment":True, "shear_angle": 30},
                
                {"augment":True, "gamma_factor":1.5, "scale_factor": 1.2, "rot_angle": 6, "shear_angle": 6}]




# for each experiment
if num_replicates == 1:
    seed_strings = [""]
else:
    seed_strings = ["_s{}".format(x) for x in range(num_replicates)] # strings indicating which replicate it is in the experiment_name

for idx, experiment_name in enumerate(experiment_names):
    # update args
    args = copy.copy(default_args)
    

    
    if GoogleStreetView:
        args = GoogleStreetView_theme(args)
    if DescribableTextures:
        args = DescribableTextures_theme(args)
    if Mias:
        args = Mias_theme(args)
    if probabilistic_inpainting:
        args = probabilistic_inpainting_theme(args)
    if small_mask:
        args = small_mask_theme(args)
    if large_context:
        args = large_context_theme(args)
    if autoencoder:
        args = autoencoder_theme(args)
    if autoencoder_small:
        args = autoencoder_small_theme(args)        
    if experiment_type == "AD": # load the args from the experiment that trained the model we want to use. Use that to overwrite most of the current args. (Purpose of this block is that we don't have to look up e.g. the model architecture of the model we trained, but can import from old json files)
        train_experiment_name = experiment_name.split("___")[0] # name of the experiment in which the model that we want to use for anomaly detection was trained
        anomaly_detection_experiment_name = experiment_name.split("___")[1] # name of the anomaly detection experiment
    
        train_experiment_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "configs", train_experiment_name + ".json"))
        with open(train_experiment_config_path) as f:
            args_train_experiment = json.load(f)
    
        args_to_update = {key:value for (key,value) in args_train_experiment.items() if key not in args_to_keep_from_AD_experiment}
        args.update(args_to_update)
    if experiment_type == "AD": # you need less GPUs and workers if you don't train
        args = AD_theme(args)
    if cpu: # it's important that this one is last, because it needs to overwrite num_workers and use_gpu
        args = cpu_theme(args)


    for key in update_dicts[idx].keys():
        assert key in args.keys(), "wrong parameters name"
    args.update(update_dicts[idx])

    ### Just a bunch of assertion that should capture common mistakes (though by no means all things that can go wrong):
    assert len(experiment_names) == len(update_dicts)
    assert not (autoencoder and autoencoder_small)
    assert not ((autoencoder or autoencoder_small) and (small_mask or large_context))
    assert ((args["image_padding_mode"] is None) or args["patch_mode"]) # doesn't make sense to use padding with full image mode
    
    assert Mias or GoogleStreetView or DescribableTextures or experiment_type == "AD" # At least one dataset has to be specified if the experiment type is not AD (then the dataset is inferred from the train experiment)
    assert Mias + GoogleStreetView + DescribableTextures <= 1

    
    assert args["dataset_name"] in ["MiasHealthy", "GoogleStreetView", "DescribableTextures"]
    assert args["anomaly_dataset_name"] in ["MiasPathological", "DTPathologicalIrreg1"]
    assert args["normalisation"] in ["mn0sd1", "range-11"]
    assert args["data_format"] in ["inpainting", "autoencoding"]
    assert args["measure_of_anomaly"] in ["absolute distance", "likelihood"] 
    assert args["window_aggregation_method"] in ["min", "mean", "max"]
    
    assert (type(args["translate_factor"]) in [list, tuple] and len(args["translate_factor"])==2)
    assert (type(args["patch_size"]) in [list, tuple] and len(args["patch_size"])==2)
    assert (type(args["mask_size"]) in [list, tuple] and len(args["mask_size"])==2)
    assert (type(args["AD_patch_stride"]) in [list, tuple] and len(args["mask_size"])==2)
    
    if args["scale_image"] is not None:
        assert (type(args["scale_image"]) in [list, tuple] and len(args["scale_image"])==2)
    if args["AD_margins"] is not None:
        assert (type(args["AD_margins"]) in [list, tuple] and len(args["AD_margins"])==2)

    seed_entered = args["seed"]
    for add_to_seed, seed_string in enumerate(seed_strings):
        args["seed"] = seed_entered + add_to_seed
    
        # create files        
        create_config_file(os.path.join(config_path, experiment_name+seed_string), args)
        create_shell_script(experiment_name=experiment_name+seed_string, experiment_type=experiment_type,
                            experiment_path=os.path.join(shell_script_path, experiment_name+seed_string), 
                                             partition=partition, args=args, time=time)
# =============================================================================
# #%% one parameter over a range
# ### !!!! This is not up to date, update with above example !!!!
    
# experiment_base_name = "test_script_generator_range_{}" # needs to include {}
# key_to_vary = "num_epochs"
# values = [10,30,70]
# partition = "Interactive"
# args = default_args
# time = None
# 
# 
# # Commonly used themes
# cpu = False
# colour_image = True
# 
# # other non-default parameters
# keys_to_update = ["num_epochs", "loss"]
# values_to_update = [50, "L2"]
# 
# # update other non-default parameters
# if cpu:
#     args = cpu_theme(args)
# if colour_image:
#     args = colour_image_theme(args)
# 
# for key,value in zip(keys_to_update, values_to_update):
#     assert key in args.keys(), "wrong parameters name"
#     args[key] = value
# 
# # update for cpu
# if cpu:
#     args.use_gpu = False
#     args.num_workers = 0
# 
# # iterate over parameter range:
# for value in values:
#     experiment_name = experiment_base_name.format(value)
#     args[key_to_vary] = value
#     create_config_file(os.path.join(config_path, experiment_name), args)
#     create_shell_script(experiment_name=experiment_name,
#                     experiment_path=os.path.join(shell_script_path, experiment_name), 
#                                      partition=partition, args=args, time=time)
#     
#     
# 
# =============================================================================
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



# =============================================================================
# ANTREAS VERSION: LOOK AT THIS BEFORE I NEED TO ADAPT MINE FURTHER
# 
# import os
# from collections import namedtuple
# 
# config = namedtuple('config', 'experiment_name num_epochs '
#                               'num_filters '
#                               'num_layers '
#                               'dim_reduction_type, seed')
# 
# experiment_templates_json_dir = '../experiment_config_template_files/'
# experiment_config_target_json_dir = '../experiment_config_files/'
# 
# configs_list = [config(experiment_name='exp_cnn_32_4_avg', num_epochs=15, num_filters=32, num_layers=4,
#                        dim_reduction_type='avg_pooling', seed=0),
#                 config(experiment_name='exp_cnn_64_4_avg', num_epochs=10, num_filters=64, num_layers=4,
#                        dim_reduction_type='avg_pooling', seed=0),
#                 config(experiment_name='exp_cnn_16_4_avg', num_epochs=10, num_filters=16, num_layers=4,
#                        dim_reduction_type='avg_pooling', seed=0),
#                 ]
# 
# if not os.path.exists(experiment_config_target_json_dir):
#     os.makedirs(experiment_config_target_json_dir)
# 
# def fill_template(script_text, config):
# 
#     for key, value in config.items():
#         script_text = script_text.replace('${}$'.format(key), str(value))
#     return script_text
# 
# def load_template(filepath):
#     with open(filepath, mode='r') as filereader:
#         template = filereader.read()
# 
#     return template
# 
# def write_text_to_file(text, filepath):
#     with open(filepath, mode='w') as filewrite:
#         filewrite.write(text)
# 
# 
# for subdir, dir, files in os.walk(experiment_templates_json_dir):
#     for template_file in files:
#         filepath = os.path.join(subdir, template_file)
#         for config in configs_list:
#             loaded_template_file = load_template(filepath=filepath)
#             config_dict = config._asdict()
#             config_dict['experiment_name'] = "_".join([template_file.replace(".json", ""),
#                                                        config.experiment_name])
#             cluster_script_text = fill_template(script_text=loaded_template_file,
#                                                 config=config_dict)
#             # name_customization = "_".join(str(item) for item in list(config._asdict().values()))
#             cluster_script_name = '{}/{}_{}.json'.format(experiment_config_target_json_dir, template_file.replace(".json", ""),
#                                                          config.experiment_name)
#             cluster_script_name = os.path.abspath(cluster_script_name)
#             write_text_to_file(cluster_script_text, filepath=cluster_script_name)
# =============================================================================

