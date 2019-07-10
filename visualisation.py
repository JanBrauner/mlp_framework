### visualise context encoder results

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import os
import math
import torchvision

import data_providers as data_providers
import model_architectures
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder


# parameters:
experiment_name = "CE_test_cluster"
batch_size = 5
seed= 0
set_to_visualise = "val"

def create_central_region_slice(image_size, size_central_region):
    margins = ((image_size[2]-size_central_region[0])/2, 
               (image_size[3]-size_central_region[1])/2) # size of margins in dimensions 1 and 2 (relative to the 3-D tensor) between the image borders and the patch borders
    
    central_region_slice = np.s_[:,:, 
                      math.ceil(margins[0]):math.ceil(image_size[2]-margins[0]), 
                      math.ceil(margins[1]):math.ceil(image_size[3]-margins[1])]
    return central_region_slice


def show(img, cax):
    npimg = img.numpy()
    cax.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

results_path = os.path.join("results")

# args, device = get_args()  # get arguments from command line. Run local debugging with settings as specified in CE_test

args, device = get_args(experiment_name)
args.batch_size = batch_size
args.use_gpu = False
args.num_workers = 0
args.seed = seed

model_dir = os.path.join("results", experiment_name, "saved_models")
model_list = os.listdir(model_dir)
for model_name in model_list:
    if model_name.endswith("_best"):
        best_model_name = model_name

state_dict = torch.load(f = os.path.join(model_dir, best_model_name), map_location="cpu")

# delete the .model prefix from the keys in the state dict (it was saved as a nn.DataParallel module):
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k != "network":
        new_state_dict[k] = v
new_state_dict["network"] = {}
for k, v in state_dict["network"].items():
    name = k.replace("model.", "") # remove `model.`
    name = name.replace("module.", "") # remove `module.`
    new_state_dict["network"][name] = v
state_dict = new_state_dict


#if args.augment:
#    augmentations = [transforms.RandomAffine(degrees=args.rot_angle, translate=args.translate_factor, 
#                                        scale=(1/args.scale_factor, args.scale_factor),
#                                        shear=args.shear_angle)]
#    # these augmentations are often used apparently:
##                transforms.RandomCrop(32, padding=4),
##                transforms.RandomHorizontalFlip(),
#    
#else:
augmentations = None

# set random seeds
rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(seed=args.seed)

# create datasets
train_data, val_data, test_data, num_output_classes = data_providers.create_dataset(args, augmentations, rng)

# create model
model = model_architectures.create_model(args)
model.load_state_dict(state_dict=state_dict["network"])
model.eval()

if set_to_visualise == "train":
    inputs, targets = next(iter(train_data))
elif set_to_visualise == "val":
    inputs, targets = next(iter(val_data))
outputs = model.forward(inputs)

central_region_slice = create_central_region_slice(inputs.shape, args.mask_size)


original_images = inputs.clone().detach()
original_images[central_region_slice] = targets

filled_in_images = inputs.clone().detach()
filled_in_images[central_region_slice] = outputs.detach()


grid_parameters = {
        "nrow":5, 
        "padding":10, 
        "normalize":False, 
        "range":None, 
        "scale_each":False, 
        "pad_value":0}

originals_grid = torchvision.utils.make_grid(original_images, **grid_parameters)
inputs_grid = torchvision.utils.make_grid(inputs, **grid_parameters)
outputs_grid = torchvision.utils.make_grid(filled_in_images, **grid_parameters)



fig = plt.figure()
cax = fig.add_subplot(311)
show(inputs_grid, cax)

cax = fig.add_subplot(312)
show(outputs_grid, cax)

cax = fig.add_subplot(313)
show(originals_grid, cax)