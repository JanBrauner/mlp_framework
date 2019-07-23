#%%
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import math
from data_providers import DescribableTexturesPathological
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import data_providers as data_providers
import model_architectures
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
from visualisation_utils import show

#%% visualise samples from a data set
which_set = "test"
transformer = transforms.ToTensor()
debug_mode = True
patch_size = (128,128)
mask_size = (64,64)
patch_stride = (30,30)
batch_size = 5
shuffle = True
batch_idx = 3


data = DescribableTexturesPathological(which_set, transformer, debug_mode, patch_size=patch_size, patch_stride=patch_stride, mask_size=mask_size)
data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

count = 0
for inputs, _, _, _ in data_loader:
    count += 1
    if count >= batch_idx:
        break

inputs_grid = torchvision.utils.make_grid(inputs, nrow=batch_size, padding=10)
show(inputs_grid)


image_list = data.image_list
label_im = data.get_label_image(7)
print(image_list[7])

#%% just visualise a grid of some images in a folder
batch_size = 10
target_size = (300,300)
random = False

#path = os.path.join("data","DTPathologicalIrreg1","test","images")
path = os.path.join("results","CE_DTD_random_patch_test_1","anomaly_maps")
image_names = os.listdir(path)
if random:
    image_names_to_load = np.random.choice(image_names, batch_size)
else:
    image_names_to_load = image_names[:batch_size]
    
images = []
for image_name in image_names_to_load:
    image = Image.open(os.path.join(path,image_name))
    image = image.resize(target_size)
    image = transforms.functional.to_tensor(image)
    images.append(image)

grid = torchvision.utils.make_grid(images, nrow=5, padding=10)
show(grid)

#%% visualise anomaly maps, ground truth segmentations, and original images
batch_size = 8
target_size = None # choose image size to resize all images to (for grid view). If None, no resizing happens.
random = True
seed = 2

def display_one_figure(batch_size, target_size, random, seed, index=None):
    """
    Usually, display batch_size iamges in one figure. Unless index is specified, then only display that image.
    """
    input_path = os.path.join("data","DTPathologicalIrreg1","test","images")
    label_path = os.path.join("data","DTPathologicalIrreg1","test","label_images")
    anomaly_path = os.path.join("results","anomaly_detection","CE_DTD_random_patch_test_1___AD1","anomaly_maps")
    image_names = os.listdir(anomaly_path)
    if random:
        rng = np.random.RandomState(seed=seed)
        image_names_to_load = rng.choice(image_names, batch_size)
    else:
        image_names_to_load = image_names[:batch_size]
      
    if index is not None:
        image_names_to_load = [image_names_to_load[index]]

    input_images = []
    label_images = []
    anomaly_maps = []

    for image_name in image_names_to_load:
        input_image = Image.open(os.path.join(input_path,image_name))
        if target_size is not None:
            input_image = input_image.resize(target_size)
        input_image = transforms.functional.to_tensor(input_image)
        input_images.append(input_image)
        
        label_image = Image.open(os.path.join(label_path,image_name))
        if target_size is not None:
            label_image = label_image.resize(target_size)
        label_image = transforms.functional.to_tensor(label_image)
        label_images.append(label_image)
        
        anomaly_map = Image.open(os.path.join(anomaly_path,image_name))
        if target_size is not None:
            anomaly_map = anomaly_map.resize(target_size)
        anomaly_map = transforms.functional.to_tensor(anomaly_map)
        anomaly_maps.append(anomaly_map)
    
    inputs_grid = torchvision.utils.make_grid(input_images, nrow=batch_size, padding=10, pad_value = 0.5)
    label_images_grid = torchvision.utils.make_grid(label_images, nrow=batch_size, padding=10, pad_value = 0.5)
    anomaly_maps_grid = torchvision.utils.make_grid(anomaly_maps, nrow=batch_size, padding=10, pad_value = 0.5)
    
    fig = plt.figure()
    
    cax = fig.add_subplot(311)
    show(inputs_grid,cax)
    
    cax = fig.add_subplot(312)
    show(anomaly_maps_grid,cax)
    
    cax = fig.add_subplot(313)
    show(label_images_grid,cax)

if target_size is not None: # display all in one figure
    display_one_figure(batch_size, target_size, random, seed)
else: # display batch_size separate figures
    for i in range(batch_size):
        display_one_figure(batch_size, target_size, random, seed, index=i)
