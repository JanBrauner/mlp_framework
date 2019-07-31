
import numpy as np
from PIL import Image
import pandas as pd
import os
import random
import torchvision.transforms as transforms
from shutil import copy2, rmtree
import math
from skimage import measure
import copy




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
    foreground_slice = np.s_[a0_range[0]:a0_range[1], a1_range[0]:a1_range[1]]# slice that excludes background
    processed_image = processed_image[foreground_slice]
    processed_image = Image.fromarray(processed_image)
    return processed_image, foreground_slice

alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
#%%


# parameters
threshold = 15 # threshold below which pixels are counted as background, for data preprocessing

healthy_split = [0.7, 0.25, 0.05]
several_copies_of_every_train_image = 10 # can be None, then there is only one copy, or can be an int, then that is how many copies are created. Of course, taking several patches from every image could also be handled by the data-provider, but doing it like this is just faster to code up, and the dataset is super small anyway

pathol_split = [0, 0.5, 0.5]

# paths
raw_path = os.path.abspath(os.path.join("data", "MiasRaw"))
base_path_healthy_WO_processing = os.path.abspath(os.path.join("data", "MiasHealthyWOProcessing"))
base_path_healthy = os.path.abspath(os.path.join("data", "MiasHealthy"))
base_path_pathol = os.path.abspath(os.path.join("data","MiasPathological"))
# all_images_path = os.path.abspath(os.path.join("data","Mias_all_images_processed")) # path to store all preprocessed images


## create directories and delete images already in them
#for base_path in [base_path_healthy, base_path_healthy_WO_processing]:
#    for which_set in ["train", "val", "test"]:
#        curr_dir = os.path.join(base_path, which_set)
#        if os.path.exists(curr_dir):
#            for file in os.listdir(curr_dir):
#                os.remove(os.path.join(curr_dir,file))
#        else:
#            os.makedirs(os.path.join(curr_dir))
#    
#for base_path in [base_path_pathol]:
#    for which_set in ["val", "test"]:
#        for image_type in ["images", "label_images"]:
#            curr_dir = os.path.join(base_path, which_set, image_type)
#            if os.path.exists(curr_dir):
#                for file in os.listdir(curr_dir):
#                    os.remove(os.path.join(curr_dir,file))
#            else:
#                os.makedirs(os.path.join(curr_dir))

#if os.path.exists(all_images_path):
#    rmtree(all_images_path)    
#os.makedirs(os.path.join(all_images_path, "images"))
#os.makedirs(os.path.join(all_images_path, "label_images"))


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
file_names_healthy = list(dict.fromkeys(file_names_healthy)) #making the list unique
random.shuffle(file_names_healthy) # shuffles in-place
file_names_healthy_train_set = file_names_healthy[:int(healthy_split[0]*len(file_names_healthy))]
file_names_healthy_val_set = file_names_healthy[int(healthy_split[0]*len(file_names_healthy)) : int((healthy_split[0]+healthy_split[1])*len(file_names_healthy))]
file_names_healthy_test_set = file_names_healthy[int((healthy_split[0]+healthy_split[1])*len(file_names_healthy)):]

# =============================================================================
# #%% Healthy: unprocessed
# # copy files from raw to respective folders
# if several_copies_of_every_train_image is not None:
#     file_name_extensions = alphabet[:several_copies_of_every_train_image]
# else:
#     file_name_extensions = [""]
# 
# for which_set in ["train", "val", "test"]:
#     if which_set == "train":
#         file_names = file_names_healthy_train_set
#     elif which_set == "val":
#         file_names = file_names_healthy_val_set
#     elif which_set == "test":
#         file_names = file_names_healthy_test_set
#     
#     for file_name in file_names:
#         for ext in file_name_extensions:
#             copy2(os.path.join(raw_path, file_name + ".pgm"), os.path.join(base_path_healthy_WO_processing, which_set, file_name + ext + ".pgm"))
# 
# =============================================================================
#%% Healthy: processed (binary segmenation, then only keep leargest connected component and crop image accordingly)
if several_copies_of_every_train_image is not None:
    file_name_extensions = alphabet[:several_copies_of_every_train_image] # e.g. ["a", "b", "c"]
else:
    file_name_extensions = [""]

image_size_list = []
for which_set in ["train", "val", "test"]:
    if which_set == "train":
        file_names = file_names_healthy_train_set
    elif which_set == "val":
        file_names = file_names_healthy_val_set
    elif which_set == "test":
        file_names = file_names_healthy_test_set

    for file_name in file_names:
        processed_image, _ = process_image(os.path.join(raw_path, file_name + ".pgm"), threshold=threshold)
        image_size_list.append(processed_image.size)
#        for ext in file_name_extensions:
#            processed_image.save(os.path.join(base_path_healthy, which_set, file_name) + ext + ".pgm")
#    processed_image.save(os.path.join(all_images_path, "images", file_name + ".pgm"))

min_image_size0 = 2000
min_image_size1 = 2000
for image_size0, image_size1 in image_size_list:
    min_image_size0 = min(min_image_size0, image_size0)
    min_image_size1 = min(min_image_size1, image_size1)

print("minimum size of input images in each dimension: ", str(min_image_size0), ", ", str(min_image_size1))
# =============================================================================
# #%% calculate mean and SD values for normalisation:
# train_images_healthy = os.listdir(os.path.join(base_path_healthy_WO_processing, "train"))
# 
# image_stack = np.empty((num_healthy, 1024, 1024)) # this approach only works because the set is small
# transform = transforms.ToTensor() # same transforms used in training
# 
# # sequentially load all images and add pixel values to image_stack as np.arrays
# for i,file_name in enumerate(train_images_healthy):
#     image = Image.open(os.path.join(base_path_healthy_WO_processing, "train", file_name))
#     tensor = transform(image)
#     image_stack[i,:,:] = tensor.numpy()
# 
# # calculate overall image stats
# mn = np.mean(image_stack)   
# sd = np.std(image_stack)
# 
# print("Mean and SD of all healthy training images (full image): ", mn, sd)
# 
# # calculate overall image for central region only
# def create_central_region_slice(image_size, size_central_region):
#     margins = ((image_size[1]-size_central_region[0])/2, 
#                (image_size[2]-size_central_region[1])/2) # size of margins in dimensions 1 and 2 (relative to the 3-D tensor) between the image borders and the patch borders
#     
#     central_region_slice = np.s_[:, 
#                       math.ceil(margins[0]):math.ceil(image_size[1]-margins[0]), 
#                       math.ceil(margins[1]):math.ceil(image_size[2]-margins[1])]
#     return central_region_slice
# 
# central_region_slice = create_central_region_slice((1,1024,1024), (256,256))
# 
# mn_central = np.mean(image_stack[central_region_slice]) 
# sd_central = np.std(image_stack[central_region_slice])
# print("Mean and SD of all healthy training images (central region only): ", mn_central, sd_central)
# 
# =============================================================================

##%% prepare pathological folder
## select pathological images only, and not the ones with calcification or assymmetry
#bool_idx = np.logical_or.reduce([df.class_of_abnormality.eq(x) for x in ["CIRC", "SPIC", "MISC", "ARCH"]])
#df_pathol = df.loc[bool_idx,:]
#
## split into datasets, only val and test required here
#num_pathol = df_pathol.shape[0]
#file_names_pathol = list(df_pathol.index)
#file_names_pathol = list(dict.fromkeys(file_names_pathol)) # making the list unique
#random.shuffle(file_names_pathol) # shuffles in-place
##file_names_pathol_train_set = file_names_pathol[:int(pathol_split[0]*len(file_names_pathol))]
#file_names_pathol_val_set = file_names_pathol[int(pathol_split[0]*len(file_names_pathol)) : int((pathol_split[0]+pathol_split[1])*len(file_names_pathol))]
#file_names_pathol_test_set = file_names_pathol[int((pathol_split[0]+pathol_split[1])*len(file_names_pathol)):]
#
#
#
#    
#### For every image: 
## create binary segmentation target images
## remove background from input image and binary segmentation image
## save both images
#for image_name in file_names_pathol:
#    
#    # create target image
#    bounding_box_coordinates = [df_pathol.loc[image_name,"y"],
#                                df_pathol.loc[image_name,"x"],
#                                df_pathol.loc[image_name,"radius"]] 
#    if type(bounding_box_coordinates[0]) is pd.core.series.Series: # This is to deal with the few cases where there are 2 or more rows for a single patient. The target image in this case combines all segmentation labels.
#        bounding_box_coordinates = [bounding_box_coordinates[i].values for i in range(3)]
#        masks = []
#        for i in range(len(bounding_box_coordinates[0])):
#            masks.append(create_target_image([bounding_box_coordinates[j][i] for j in range(3)]))
#        target_image = np.maximum(masks[0],masks[1])
#        if len(masks) > 2:
#            for mask in masks[2:len(masks)+1]:
#                target_image = np.maximum(target_image, mask)
#    else:
#        target_image = create_target_image(bounding_box_coordinates)
#        
#    # remove background
#    processed_input_image, foreground_slice = process_image(os.path.join(raw_path, image_name + ".pgm"), threshold=threshold)
#    processed_target_image = target_image[foreground_slice] 
#
#    # saving into respective sets
#    if image_name in file_names_pathol_val_set:
#        which_set = "val"
#    elif image_name in file_names_pathol_test_set:
#        which_set = "test"
#    else:
#        raise Exception
#        
#    processed_input_image.save(os.path.join(base_path_pathol, which_set, "images", image_name + ".pgm"))
##    processed_input_image.save(os.path.join(all_images_path, "images", image_name + ".pgm"))
#
#    processed_target_image = Image.fromarray(np.uint8(processed_target_image*255))
#    processed_target_image.save(os.path.join(base_path_pathol, which_set, "label_images",image_name) + ".pgm")
##    processed_target_image.save(os.path.join(all_images_path, "label_images", image_name + ".pgm"))