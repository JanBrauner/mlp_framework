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
experiment_name = "CE_random_patch_location_test_1"
batch_size = 5
image_batch_idx = 3 # use different number to see different images
set_to_visualise = "train"

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
    iterator = iter(train_data)
elif set_to_visualise == "val":
    iterator = iter(val_data)

for i in range(image_batch_idx): # show the n-th batch
    inputs, targets = next(iterator)

outputs = model.forward(inputs)

# inverse normalization
if args.normalisation == "mn0sd1":
    if args.patch_location_during_training == "central" and args.dataset_name == "MiasHealthy":
        mn = 0.39865 
        SD = 0.30890
    elif args.patch_location_during_training == "random" and args.dataset_name == "MiasHealthy":
        mn = 0.14581
        SD = 0.25929
elif args.normalisation == "range-11":
    mn = 0.5
    SD = 0.5

inv_normalize = transforms.Normalize((-mn/SD,), (1./SD,))

for images in [inputs,targets,outputs]:
    for idx,image in enumerate(images): # since these are a batch of images, but transforms work on indivdual images
        images[idx,:,:,:] = inv_normalize(image)




central_region_slice = create_central_region_slice(inputs.shape, args.mask_size)


original_images = inputs.clone().detach()
original_images[central_region_slice] = targets

filled_in_images = inputs.clone().detach()
filled_in_images[central_region_slice] = outputs.detach()


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



fig = plt.figure()
cax = fig.add_subplot(311)
show(inputs_grid, cax)

cax = fig.add_subplot(312)
show(outputs_grid, cax)

cax = fig.add_subplot(313)
show(originals_grid, cax)
