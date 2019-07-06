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
from shutil import copy2

# parameters
healthy_split = [0.7, 0.3, 0.0]
pathol_split = [0, 0, 1]

# paths
raw_path = os.path.join("data", "MiasRaw")
base_path_healthy = os.path.join("data", "MiasHealthy")
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
file_names_healthy_train_set = file_names_healthy[0 : int(healthy_split[0]*len(file_names_healthy))]
file_names_healthy_val_set = file_names_healthy[int(healthy_split[0]*len(file_names_healthy)) : len(file_names_healthy)] # no test set necessary here

# copy files from raw to respective folders
for file_name in file_names_healthy_train_set:
    copy2(os.path.join(raw_path, file_name + ".pgm"), os.path.join(base_path_healthy, "train"))    

for file_name in file_names_healthy_val_set:
    copy2(os.path.join(raw_path, file_name + ".pgm"), os.path.join(base_path_healthy, "val"))

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


            
