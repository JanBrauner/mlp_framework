# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:49:28 2019

@author: MC JB
"""
import numpy as np
from PIL import Image
import pandas as pd
import os
import random
import torchvision.transforms as transforms
from shutil import copy2
import math



# parameters
healthy_split = [0.7, 0.25, 0.05]
threshold = 15 # threshold below which pixels are counted as background, for data preprocessing

pathol_split = [0, 0, 1]

# paths
raw_path = os.path.join("data", "MiasRaw")
base_path_healthy = os.path.join("data", "MiasHealthy")
base_path_healthy_processed = os.path.join("data", "MiasHealthyProcessed")
base_path_pathol = os.path.join("data","MiasPathological")


# =============================================================================
# image_path = os.path.join(base_path, which_set, "images")
# target_image_path = os.path.join(base_path, which_set, "target_images")
# image_list = os.listdir(image_path)
# image_names = [os.path.splitext(image_list[i])[0] for i in range(len(image_list))] # remove file extension for comparing with name in info table
# =============================================================================


def create_circular_mask(h, w, center=None, radius=None):

    X, Y = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def create_target_image(bounding_box_coordinates):
    bounding_box_coordinates[0] = 1023 - bounding_box_coordinates[0] # coordinate origin is bottom left corner
    if not np.isnan(bounding_box_coordinates[0]):
        target_image = create_circular_mask(1024, 1024, center=bounding_box_coordinates[0:2], radius=bounding_box_coordinates[2])
    else:
        target_image = np.zeros((1024,1024))
    return target_image



random.seed(0)

# load data table
header = ["character_background_tissue", "class_of_abnormality", "severity", "x", "y", "radius"] # only 6 elements since the first row is the index
df = pd.read_csv(os.path.join(raw_path, "table.txt"), sep=" ", header=None, names=header, index_col=0)
df = df[df.x != "*NOTE"] # delete three images that have widely dispersed calcifications

# after having deleted the outlier columns, convert dtypes. Needs to be float, not int, to account for nan
df.loc[:,"x"] = df.loc[:,"x"].astype(float)
df.loc[:,"y"] = df.loc[:,"y"].astype(float)
df.loc[:,"y"] = df.loc[:,"y"].astype(float)

#%% prepare healthy image folders
# select healthy images only
df_healthy = df[df.class_of_abnormality == "NORM"]

# split healthy images into sets
num_healthy = df_healthy.shape[0]
file_names_healthy = list(df_healthy.index)
random.shuffle(file_names_healthy) # shuffles in-place
file_names_healthy_train_set = file_names_healthy[:int(healthy_split[0]*len(file_names_healthy))]
file_names_healthy_val_set = file_names_healthy[int(healthy_split[0]*len(file_names_healthy)) : int((healthy_split[0]+healthy_split[1])*len(file_names_healthy))]
file_names_healthy_test_set = file_names_healthy[int((healthy_split[0]+healthy_split[1])*len(file_names_healthy)):]

#%% Healthy: unprocessed
# copy files from raw to respective folders
for file_name in file_names_healthy_train_set:
    copy2(os.path.join(raw_path, file_name + ".pgm"), os.path.join(base_path_healthy, "train"))    

for file_name in file_names_healthy_val_set:
    copy2(os.path.join(raw_path, file_name + ".pgm"), os.path.join(base_path_healthy, "val"))
    
for file_name in file_names_healthy_test_set:
    copy2(os.path.join(raw_path, file_name + ".pgm"), os.path.join(base_path_healthy, "test"))


#%% Healthy: processed (binary segmenation, then only keep leargest connected component and crop image accordingly)
from skimage import measure
import copy

def get_largest_CC(connected_components):
    # get largest connected component that isn't zero
    largest_CC = np.argmax(np.bincount(connected_components.flat)[1:]) + 1
    return largest_CC

def process_image(image_path, threshold):
    # load image and convert to numpy array
    image = Image.open(image_path)
    image = np.array(image)    
    
    # binary image is zero if original image is <= threshold, 1 otherwise
    binary_image = copy.copy(image)
    binary_image[binary_image <= threshold] = 0
    binary_image[binary_image != 0] = 1
    
    # determine connected components
    connected_components = measure.label(binary_image)
    largest_CC = get_largest_CC(connected_components)
    
    # only keep that part of original image that corresponds to the largest connected component
    # Do this by setting every pixel that doesn't belong to the largest connected component to zero, and discard all rows and columns that only contain zeros
    processed_image = (connected_components==largest_CC)*image
    col_sums = np.sum(processed_image, axis=0)
    row_sums = np.sum(processed_image, axis=1)
    a0_range = (np.min(np.where(row_sums > 0)), np.max(np.where(row_sums > 0)))
    a1_range = (np.min(np.where(col_sums > 0)), np.max(np.where(col_sums > 0)))
    processed_image = processed_image[a0_range[0]:a0_range[1], a1_range[0]:a1_range[1]]
    processed_image = Image.fromarray(processed_image)
    return processed_image




for file_name in file_names_healthy_train_set:
    processed_image = process_image(os.path.join(raw_path, file_name + ".pgm"), threshold=threshold)
    processed_image.save(os.path.join(base_path_healthy_processed, "train", file_name) + ".pgm")

for file_name in file_names_healthy_val_set:
    processed_image = process_image(os.path.join(raw_path, file_name + ".pgm"), threshold=threshold)
    processed_image.save(os.path.join(base_path_healthy_processed, "val", file_name) + ".pgm")
    
for file_name in file_names_healthy_test_set:
    processed_image = process_image(os.path.join(raw_path, file_name + ".pgm"), threshold=threshold)
    processed_image.save(os.path.join(base_path_healthy_processed, "test", file_name) + ".pgm")


#%% calculate mean and SD values for normalisation:
train_images_healthy = os.listdir(os.path.join(base_path_healthy, "train"))

image_stack = np.empty((num_healthy, 1024, 1024)) # this approach only works because the set is small
transform = transforms.ToTensor() # same transforms used in training

# sequentially load all images and add pixel values to image_stack as np.arrays
for i,file_name in enumerate(train_images_healthy):
    image = Image.open(os.path.join(base_path_healthy, "train", file_name))
    tensor = transform(image)
    image_stack[i,:,:] = tensor.numpy()

# calculate overall image stats
mn = np.mean(image_stack)   
sd = np.std(image_stack)

print("Mean and SD of all healthy training images (full image): ", mn, sd)

# calculate overall image for central region only
def create_central_region_slice(image_size, size_central_region):
    margins = ((image_size[1]-size_central_region[0])/2, 
               (image_size[2]-size_central_region[1])/2) # size of margins in dimensions 1 and 2 (relative to the 3-D tensor) between the image borders and the patch borders
    
    central_region_slice = np.s_[:, 
                      math.ceil(margins[0]):math.ceil(image_size[1]-margins[0]), 
                      math.ceil(margins[1]):math.ceil(image_size[2]-margins[1])]
    return central_region_slice

central_region_slice = create_central_region_slice((1,1024,1024), (256,256))

mn_central = np.mean(image_stack[central_region_slice]) 
sd_central = np.std(image_stack[central_region_slice])
print("Mean and SD of all healthy training images (central region only): ", mn_central, sd_central)


#%% prepare pathological folder
# select pathological images only
df_pathol = df[df.class_of_abnormality != "NORM"]

# copy files of input images to test folder (only test set here)
for file_name in df_pathol.index:
    copy2(os.path.join(raw_path, file_name + ".pgm"), os.path.join(base_path_pathol, "test", "images"))
    
# create and save binary segmentation target images
for image_name in df_pathol.index:
    bounding_box_coordinates = [df_pathol.loc[image_name,"y"],
                                df_pathol.loc[image_name,"x"],
                                df_pathol.loc[image_name,"radius"]] 
    if type(bounding_box_coordinates[0]) is pd.core.series.Series: # This is to deal with the few cases where there are 2 or more rows for a single patient. The target image in this case combines all segmentation labels.
        bounding_box_coordinates = [bounding_box_coordinates[i].values for i in range(3)]
        masks = []
        for i in range(len(bounding_box_coordinates[0])):
            masks.append(create_target_image([bounding_box_coordinates[j][i] for j in range(3)]))
        target_image = np.maximum(masks[0],masks[1])
        if len(masks) > 2:
            for mask in masks[2:len(masks)+1]:
                target_image = np.maximum(target_image, mask)
    else:
        target_image = create_target_image(bounding_box_coordinates)
        
    target_image = Image.fromarray(np.uint8(target_image*255))
    target_image.save(os.path.join(base_path_pathol, "test", "target_images",image_name) + ".pgm")

