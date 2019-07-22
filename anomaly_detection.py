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
train_experiment_name = args.experiment_name.split("___")[0] # name of the experiment in which the model that we want to use for anomaly detection was trained
anomaly_detection_experiment_name = args.experiment_name.split("___")[1] # name of the anomaly detection experiment

args_train_experiment, _ = get_args(train_experiment_name)
args_to_update = {key:value for (key,value) in args_train_experiment.items() if key not in args_to_keep_from_AD_experiment}
args.update(args_to_update)

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

experiment = AnomalyDetectionExperiment(experiment_name=train_experiment_name, 
                                        anomaly_detection_experiment_name=anomaly_detection_experiment_name,
                                        model=model, 
                                        device=device,
                                        test_data_loader=test_data_loader, 
                                        test_dataset=test_dataset,
                                        measure_of_anomaly=args.measure_of_anomaly, 
                                        window_aggregation_method=args.window_aggregation_method, 
                                        save_anomaly_maps=args.save_anomaly_maps,
                                        is_gpu = args.is_gpu)

experiment.run_experiment()
