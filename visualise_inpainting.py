"""
Visualise context encoder results:
    Load a trained model, and visualise inputs, outputs, original images, ...

"""

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

#%%
# parameters:
experiment_name = "CE_DTD_random_patch_test_1" #  "CE_DTD_random_patch_test_1"
batch_size = 8 # number of images per row
image_batch_idx = 0 # use different number to see different images
seed = 0 # to see different regions of the images
set_to_visualise = "train"
force_patch_location = "central" # "False": every model gets visualised with patches from the location it was trained with. Otherwise, specify the patch_location the models should be tested with
force_dataset = False # "False": every model gets visualised with dataset it was trained. Otherwise, specify the dataset the models should be tested with. !Of course, you can't force a model that was trained on gray-scale images to work on RGB images

# add new parameters to older experiments that were run when that argument didn't yet exist and thus don't hove that argument in their config files
patch_mode_default = True

# paths
results_path = os.path.join("results")
model_dir = os.path.join("results", experiment_name, "saved_models")


#%%
def create_central_region_slice(image_size, size_central_region):
    # create slice of the central region of an image (dimensions (CxHxW)), when the size of the central region is central_region_size (HxW)
    margins = ((image_size[2]-size_central_region[0])/2, 
               (image_size[3]-size_central_region[1])/2) # size of margins in dimensions 1 and 2 (relative to the 3-D tensor) between the image borders and the patch borders
    
    central_region_slice = np.s_[:,:, 
                      math.ceil(margins[0]):math.ceil(image_size[2]-margins[0]), 
                      math.ceil(margins[1]):math.ceil(image_size[3]-margins[1])]
    return central_region_slice



def update_state_dict_keys(state_dict):
    # Modify keys in a model state dict to use a model that was serialised as a nn.DataParallel module:
    # delete the .model prefix from the keys in the state dict
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
    return new_state_dict


#%%
### update args
args, device = get_args(experiment_name)
args.batch_size = batch_size # to display a specified amount of image per figure
args.use_gpu = False # to run on cpu
args.num_workers = 0 # to run on cpu
args.seed = seed # to ensure the same images are shown for different models

# for parameters specified above
if force_patch_location:
    print("Force patch location to {}".format(force_patch_location))
    args.patch_location_during_training = force_patch_location 

if force_dataset:
    print("Force dataset to {}".format(force_dataset))
    args.dataset_name = force_dataset

# add new parameters to old experiments for compatibility:
try:
    args.patch_mode
except: # exception is thrown if args has no such attribute
    args_to_update = {"patch_mode": patch_mode_default}
    args.__dict__.update(args_to_update)


### Load model from  best epoch of that experiment
# find best model
model_list = os.listdir(model_dir)
for model_name in model_list:
    if model_name.endswith("_best"):
        best_model_name = model_name

# load best model's state dict
state_dict = torch.load(f = os.path.join(model_dir, best_model_name), map_location="cpu")
state_dict = update_state_dict_keys(state_dict)

# create model
model = model_architectures.create_model(args)
model.load_state_dict(state_dict=state_dict["network"])
model.eval()


### Create data
if args.augment:
    augmentations = [transforms.RandomAffine(degrees=args.rot_angle, translate=args.translate_factor, 
                                        scale=(1/args.scale_factor, args.scale_factor),
                                        shear=args.shear_angle)]
    # these augmentations are often used apparently:
#                transforms.RandomCrop(32, padding=4),
#                transforms.RandomHorizontalFlip(),
    
else:
    augmentations = None

# set random seeds
rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(seed=args.seed)

# create datasets
args.batch_size = 100 # this is just for a super weird way to randomise the order of images shown in train and val set, without having to adjust the code for the data provider. I first draw a batch with batchsize 100, and then randomly choose "batch_size" from it. See (*1) below
train_data, val_data, test_data, num_output_classes = data_providers.create_dataset(args, augmentations, rng)
args.batch_size = batch_size

# load batch for visualisation
if set_to_visualise == "train":
    data_loader = train_data
elif set_to_visualise == "val":
    data_loader = val_data
elif set_to_visualise == "test":
    data_loader = test_data

# load n-th batch, n = image_batch_idx
cnt = 0
for inputs, targets in data_loader:
    cnt += 1
    if cnt >= image_batch_idx:
        break

# randomise images shown (*1)
image_idx = rng.choice(100, args.batch_size, replace=False)
inputs = inputs[image_idx,:,:,:]
targets = targets[image_idx,:,:,:]

# create original images by combining inputs (masked out) and targets
central_region_slice = create_central_region_slice(inputs.shape, args.mask_size)
original_images = inputs.clone().detach()
original_images[central_region_slice] = targets



### Send data through model and create outputs
outputs = model.forward(inputs)

filled_in_images = inputs.clone().detach()
filled_in_images[central_region_slice] = outputs.detach()


### inverse normalization of all images
if args.dataset_name == "MiasHealthy":
    if args.normalisation == "mn0sd1":
        if args.patch_location_during_training == "central":
            mn = (0.39865,) 
            SD = (0.30890,)
        elif args.patch_location_during_training == "random":
            mn = (0.14581,)
            SD = (0.25929,)
    elif args.normalisation == "range-11":
        mn = (0.5,)
        SD = (0.5,)

elif args.dataset_name == "GoogleStreetView":
    if args.normalisation == "mn0sd1":
        raise NotImplementedError
    elif args.normalisation == "range-11":
        mn = [0.5, 0.5, 0.5]
        SD = [0.5, 0.5, 0.5]

elif args.dataset_name == "DescribableTextures":
    if args.normalisation == "mn0sd1":
        raise NotImplementedError
    elif args.normalisation == "range-11":
        mn = [0.5, 0.5, 0.5]
        SD = [0.5, 0.5, 0.5]
        
neg_of_mn_over_SD = [-x/y for x,y in zip(mn,SD)] # this has a bit weird syntax because it needs to deal with tupels and lists, since that's what transforms.normalise expects as input
one_over_SD = [1./x for x in SD]
inv_normalize = transforms.Normalize(neg_of_mn_over_SD, one_over_SD)

for images in [inputs,filled_in_images,original_images]:
    for idx,image in enumerate(images): # since these are a batch of images, but transforms work on indivdual images
        images[idx,:,:,:] = inv_normalize(image)


#%% Display images
### create image grid
grid_parameters = {
        "nrow":args.batch_size, 
        "padding":10, 
        "normalize":False, 
        "range":None, 
        "scale_each":False, 
        "pad_value":0}

originals_grid = torchvision.utils.make_grid(original_images, **grid_parameters)
inputs_grid = torchvision.utils.make_grid(inputs, **grid_parameters)
outputs_grid = torchvision.utils.make_grid(filled_in_images, **grid_parameters)


### plot images
fig = plt.figure()
cax = fig.add_subplot(311)
show(inputs_grid, cax)

cax = fig.add_subplot(312)
show(outputs_grid, cax)

cax = fig.add_subplot(313)
show(originals_grid, cax)
