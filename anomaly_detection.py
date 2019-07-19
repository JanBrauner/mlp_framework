"""
ToDo:
   Current ToDo: create script generator and args and experiment_name in a way that makes sense. Current problem is the model is not build with the parameters specified in the original experiment. That could probably be fixed by just loading the dict or original_experiment_name
   Then test and debug the whole thing
    
    
    
    Fix the load_best_model thing with the is_gpu flag
    If I have to do stuff on the cluster, I might have to send stuff to the GPU, detach at the right times, ...
    
    At the end, refactor DataSet nicely
    enable combining heatmaps from several models?
  
    do some testing
        DataSet works
        normalisation_map works
        Let's just first see if it does what I think it does. If it does, then no more testing is necessary
      
    Script generator: simply have an option to do trainig, anomaly detection, or both
    
    Pipeline/script generation: probably a good solution would be to have anomaly_detection_experimemt_names
        Think about this for a second, how this would be nice to analyse as well!
        
    I need to save the AUC scores somewhere
    Maybe include further scores
            
    Script generator: include all the new args and update the description of the args that have double meaning
    
    Defaults:
        measure_of_anomaly = "absolute distance"
        window_aggregation_method = "mean"
        save_anomaly_maps = True
At the end:
    
    
Notes:
    - write everything to be [BxCxHxW]-compatible. I might want to do it on the cluster.
"""



### prepare
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision
import torch.nn as nn
import numpy as np
import math
from data_providers import DescribableTexturesPathological
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import tqdm

import data_providers as data_providers
import model_architectures
from arg_extractor import get_args
from experiment_builder import AnomalyDetectionExperiment
from misc_utils import get_aucroc
from storage_utils import load_best_model_state_dict, save_statistics


def create_central_region_slice(image_size, size_central_region):
    # create slice of the central region of an image (dimensions (CxHxW)), when the size of the central region is central_region_size (HxW)
    margins = ((image_size[2]-size_central_region[0])/2, 
               (image_size[3]-size_central_region[1])/2) # size of margins in dimensions 1 and 2 (relative to the 3-D tensor) between the image borders and the patch borders
    
    central_region_slice = np.s_[:,:, 
                      math.ceil(margins[0]):math.ceil(image_size[2]-margins[0]), 
                      math.ceil(margins[1]):math.ceil(image_size[3]-margins[1])]
    return central_region_slice



args, device = get_args("CE_DTD_random_patch_test_1")  # get arguments from command line/json config.
original_experiment_name = args.experiment_name.split("--")[0] # name of the experiment in which the model that we want to use for anomaly detection was trained
anomaly_detection_experiment_name = args.experiment_name.split("--")[1] # name of the anomaly detection experiment

# temp for local debugging:
args.use_gpu = False
args.num_workers = 0


# set random seeds
rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(seed=args.seed)

# create dataset:
test_data_loader, test_dataset = data_providers.create_dataset_with_anomalies(
        anomaly_dataset_name=args.anomaly_dataset_name, which_set="test", 
        normalisation=args.normalisation, batch_size=args.AD_batch_size, 
        patch_size=args.patch_size, patch_stride=args.AD_patch_stride, mask_size=args.mask_size, 
        num_workers=args.num_workers, debug_mode=args.debug_mode)


# create model
model = model_architectures.create_model(args)



#### Probably here the class begins!
# build experiment

# run experiment


# paths:

experiment = AnomalyDetectionExperiment(experiment_name=original_experiment_name, 
                                        anomaly_detection_experiment_name=anomaly_detection_experiment_name,
                                        model=model, 
                                        device=device,
                                        test_data_loader=test_data_loader, 
                                        test_dataset=test_dataset,
                                        measure_of_anomaly=args.measure_of_anomaly, 
                                        window_aggregation_method=args.window_aggregation_method, 
                                        save_anomaly_maps=args.save_anomaly_maps)

experiment.run_experiment()
