# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:49:28 2019

@author: MC JB
"""
import numpy as np
from PIL import Image
import pandas as pd
import os

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


which_set = "test"

# paths
base_path = os.path.join("data","pathological_1")
image_path = os.path.join(base_path, which_set, "images")
target_image_path = os.path.join(base_path, which_set, "target_images")
image_list = os.listdir(image_path)
image_names = [os.path.splitext(image_list[i])[0] for i in range(len(image_list))] # remove file extension for comparing with name in info table

# load data
header = ["reference no", "character background tissue", "class of abnomarlity", "severity", "x of centre", "y of centre", "radius"]
info = pd.read_csv(os.path.join(base_path, "table.txt"), sep=" ", header=None, names=header, index_col=0)


# create and save binary segmentation target images
for image_name in image_names:
    bounding_box_coordinates = [info.loc[image_name,"y of centre"],
                                info.loc[image_name,"x of centre"],
                                info.loc[image_name,"radius"]] 
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
    target_image.save(os.path.join(target_image_path,image_name) + ".pgm")


            