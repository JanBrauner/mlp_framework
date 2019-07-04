# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 19:40:19 2019

@author: MC JB
"""
#%% Testing the XXXHealthyDataset
from data_providers import XXXHealthyDataset
import torch
import torchvision.transforms as transforms
import numpy as np
import math

task = "regression"
debug_mode=False 
patch_size=(1024,1024)
patch_location="central"
mask_size=(64,64)


transform = True
gamma_factor=1
rot_angle=30
shear_angle=30
translate_distance=(0.2,0.2)
scale_factor=1.5

# =============================================================================
# standard_transforms = [transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
# =============================================================================

standard_transforms = [transforms.ToTensor()]

if transform:
    augmentations = [transforms.RandomAffine(degrees=rot_angle, translate=translate_distance, 
                                        scale=(1/scale_factor, scale_factor),
                                        shear=shear_angle)]
    transformer_train = transforms.Compose(augmentations + standard_transforms)
else:
    transformer_train = transforms.Compose(standard_transforms)
    
trainset = XXXHealthyDataset(which_set="train", task=task,
                 transformer=transformer_train,
                 debug_mode=debug_mode, 
                 patch_size=patch_size, patch_location=patch_location, mask_size=mask_size)

trainset_iter = iter(trainset)
for iterations in range(np.random.randint(1,100)):
    data = next(trainset_iter)


def create_central_region_slice(image_size, size_central_region):
    margins = ((image_size[1]-size_central_region[0])/2, 
               (image_size[2]-size_central_region[1])/2) # size of margins in dimensions 1 and 2 (relative to the 3-D tensor) between the image borders and the patch borders
    
    central_region_slice = np.s_[:, 
                      math.ceil(margins[0]):math.ceil(image_size[1]-margins[0]), 
                      math.ceil(margins[1]):math.ceil(image_size[2]-margins[1])]
    return central_region_slice

# tests    
assert type(data[0]) is torch.Tensor
assert type(data[1]) is torch.Tensor
assert data[0].size() == (1,) + patch_size
assert data[1].size() == (1,) + mask_size

central_region_slice = create_central_region_slice(data[0].size(), mask_size)
assert torch.all(torch.eq(data[0][central_region_slice], 0))# torch.all(torch.eq(data[0][central_region_slice], data[1]))

# visualisation
composed = data[0].clone()
composed[central_region_slice] = data[1]
simple_back_transformer = transforms.ToPILImage()
image = simple_back_transformer(data[0])
target = simple_back_transformer(data[1])
composed = simple_back_transformer(composed)
image.show()
target.show()
composed.show()
