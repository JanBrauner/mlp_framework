"""
Visualise outputs of trained models.
E.g.: context encoder results:
    Load a trained model, and visualise inputs, outputs (as filled in images), original images, ...
E.g.: Autoencoder
    Load a trained model, and visualise inputs, outputs,

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import os
import math
import torchvision
from ast import literal_eval

import data_providers as data_providers
import model_architectures
from arg_extractor import get_args
from visualisation_utils import image_grid_with_groups
from storage_utils import load_best_model_state_dict
from misc_utils import create_central_region_slice

#%%
# parameters:
experiment_name = "r7_CE_Mias_prob_scale_0p35" #"r4_CE_Mias_augtest_best_combo_s2" # "r6_CE_Mias_padding_const_s1" # "r4_CE_Mias_augtest_best_combo_s0" #  # "r4_CE_Mias_augtest_best_combo_s2"
save_image = True
save_dir = "C:\\Users\\MC JB\\Dropbox\\dt\\Edinburgh\\project\\final report\\figures\\"
save_name = "MIAS_best_stand_AD_model_inpainting.png"
save_path = save_dir + save_name
batch_size = 8 # number of images per row
image_batch_idx = 0 # use different number to see different images
seed = 1 # to see different regions of the images
set_to_visualise = "test"
force_patch_location = "central" # "False": every model gets visualised with patches from the location it was trained with. Otherwise, specify the patch_location the models should be tested with
force_dataset = False # "False": every model gets visualised with dataset it was trained. Otherwise, specify the dataset the models should be tested with. !Of course, you can't force a model that was trained on gray-scale images to work on RGB images

# add new parameters to older experiments that were run when that argument didn't yet exist and thus don't hove that argument in their config files
default_args = {
"patch_mode" : True,
"scale_image" : None,
"data_format": "inpainting",
"image_padding_mode": None}

# paths
results_path = os.path.join("results")
model_dir = os.path.join("results", experiment_name, "saved_models")


#%%
### get args (for model and dataset) from config, update args as specifified above
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
for key,value in default_args.items():
    if key not in args.__dict__.keys():
        args.__dict__.update({key: value})




#%%
### Load model from  best epoch of that experiment
state_dict = load_best_model_state_dict(model_dir=model_dir, use_gpu=False, saved_as_parallel_load_as_single_process=True, saved_whole_module_load_only_model=True)

# create model
model = model_architectures.create_model(args)
model.load_state_dict(state_dict=state_dict["network"])
model.eval()





#%%
### Create data
if args.augment:
    # create custom gamma adjustment augmentation
    
    GammaAdjustment = transforms.Lambda(lambda img: transforms.functional.adjust_gamma(img, gamma=args.gamma_factor**np.random.uniform(-1,1)))

    augmentations = [transforms.RandomHorizontalFlip(),
                     GammaAdjustment,
                     transforms.RandomAffine(degrees=args.rot_angle, 
                                             translate=args.translate_factor, 
                                             scale=(1/args.scale_factor, args.scale_factor),
                                             shear=args.shear_angle)]
else:
    augmentations = None

# set random seeds
rng = np.random.RandomState(seed=args.seed)
st0 = rng.get_state()
torch.manual_seed(seed=args.seed)


# create datasets
args.batch_size = 500 # this is just for a super weird way to randomise the order of images shown in train and val set, without having to adjust the code for the data provider. I first draw a batch with batchsize 100, and then randomly choose "batch_size" from it. See (*1) below
train_data, val_data, test_data, num_output_classes = data_providers.create_dataset(args, augmentations, rng, precompute_patches=False)
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
rng.set_state(st0)
image_idx = rng.choice(inputs.shape[0], args.batch_size, replace=False)
inputs = inputs[image_idx,:,:,:]
targets = targets[image_idx,:,:,:]





#%%
### Send data through model and create outputs
outputs = model.forward(inputs)
if args.task == "classification": # this is only the case if we trained a probabilistic CE, at least atm.
    # in this case, outputs has shape B x classes x channel x H x W
    _, outputs = torch.max(outputs, dim=1)  # get argmax of predictions






#%%
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

if args.task == "regression": # all images are within range [-1,1], or have zero mean and unit variance. So all images need to go through inverse normalisation
    for images in [inputs,outputs,targets]:
        for idx,image in enumerate(images): # since these are a batch of images, but transforms work on indivdual images
            images[idx,:,:,:] = inv_normalize(image)
elif args.task == "classification": # inputs are within range [-1,1], or have zero mean and unit variance. But outputs and targets are in [0,255]
    for idx,image in enumerate(inputs): # since these are a batch of images, but transforms work on indivdual images
        inputs[idx,:,:,:] = inv_normalize(image)*255 # first bring to range [0,1], then to range [0,255]
    inputs = inputs.type(torch.uint8)




#%%
### create composition images
if args.data_format == "inpainting":
    # create original images by combining inputs (masked out) and targets
    central_region_slice = create_central_region_slice(inputs.shape[2:], args.mask_size)
    central_region_slice = np.s_[:, :, central_region_slice[0], central_region_slice[1]]
    original_images = inputs.clone().detach()
    original_images[central_region_slice] = targets
    
    filled_in_images = inputs.clone().detach()
    filled_in_images[central_region_slice] = outputs.detach()
    
    
elif args.data_format == "autoencoding":
    pass




#%%
### plot
if args.task == "regression":
    pad_value = 1
elif args.task == "classification":
    pad_value = 255
grid_parameters = {
        "nrow":args.batch_size, 
        "padding":5, 
        "normalize":False, 
        "range":None, 
        "scale_each":False, 
        "pad_value":pad_value}

if args.data_format == "inpainting":
    fig = image_grid_with_groups(inputs, original_images, filled_in_images, grid_parameters=grid_parameters)
elif args.data_format == "autoencoding":
    fig = image_grid_with_groups(inputs, outputs, grid_parameters=grid_parameters)

if save_image:
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
#    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=300)
