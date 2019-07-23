"""
Start with all images in data/DescribableTextures/raw (inside their subdirectories, as provided by the dataset)
Also needs the files train1.txt, val1.txt, test1.txt, which describe a 1/3-1/3-1/3 split by the dataset providers
"""

import numpy as np
from PIL import Image
import pandas as pd
import os
import random
import torchvision.transforms as transforms
from shutil import copy2
import math
import matplotlib.pyplot as plt

import mask_utils
from torchvision.utils import make_grid
from visualisation_utils import show
from skimage.transform import resize
# 
def create_list_of_images(file_path):
    # create list of images from file
    with open(file_path) as f:
        string = f.read()
        list_of_images = string.split("\n")
        list_of_images = [x for x in list_of_images if x != ""]
        return list_of_images


# paths
raw_data_path = os.path.join("data","DescribableTextures", "raw")
image_base_path = os.path.join("data","DescribableTextures")
pathol_image_base_path = os.path.join("data","DTPathologicalIrreg1")

# parameters
test_images_per_category = 20

# =============================================================================
# #%% distribute images to the respective folders
# 
# # 2/3rd of of images go to train set: copy images from train1.txt and test1.txt to the train subdirectory
# #file_path = os.path.join(raw_data_path, "train1.txt")
# #list_of_images = create_list_of_images(file_path)
# list_of_images1 = create_list_of_images(os.path.join(raw_data_path, "train1.txt"))
# list_of_images2 = create_list_of_images(os.path.join(raw_data_path, "test1.txt"))
# list_of_images = list_of_images1 + list_of_images2
# 
# # delete all files in resepctive folders:
# for which_set in ["train","val","test"]:
#     for file_name in os.listdir(os.path.join(image_base_path, which_set)):
#         os.remove(os.path.join(image_base_path, which_set, file_name))
# 
# for image_name in list_of_images:
#     copy2(os.path.join(raw_data_path, image_name), os.path.join(image_base_path, "train"))
#     
# # The remaining images are split between val and test: copy images from val1.txt to the validation subdirectory
# list_of_images = create_list_of_images(os.path.join(raw_data_path, "val1.txt"))
# for idx, image_name in enumerate(list_of_images):
#     if idx % (40/test_images_per_category) == 0: # 40 imges of each category in that file
#         copy2(os.path.join(raw_data_path, image_name), os.path.join(image_base_path, "test"))
#     else:
#         copy2(os.path.join(raw_data_path, image_name), os.path.join(image_base_path, "val"))
# 
# 
# =============================================================================
#%% Create pathological version of dataset
"""
- Take the test images
- For every test image, paste 1 or 2 randomly shaped regions from a different randomly test image selected image into the first image as "lesions"
- save the mask of the random regions as "ground truth labels"
"""

list_of_test_images = os.listdir(os.path.join(image_base_path, "test"))



# create new folder and delete all files in respective folders if the already exist:
for which_set in ["val", "test"]:    
    for image_type in ["images","label_images"]:
        try:
            os.makedirs(os.path.join(pathol_image_base_path, which_set, image_type))
        except FileExistsError:
            for file_name in os.listdir(os.path.join(pathol_image_base_path, which_set, image_type)):
                os.remove(os.path.join(pathol_image_base_path, which_set, image_type, file_name))


# modify each test image from test folder of the normal DT dataset and copy the modfied version to the new folder. Also save mask.
mask_sizes = []
for count, image_name in enumerate(list_of_test_images):
    
    # open image
    image = Image.open(os.path.join(image_base_path,"test",image_name))
    
    # create mask
    npimage = np.array(image) # 
    mean_image_dim = (npimage.shape[0]+npimage.shape[1])/2
    mask, current_mask_sizes = mask_utils.batch_multi_random_blobs(img_size=(npimage.shape[2], npimage.shape[0], npimage.shape[1]), max_num_blobs=2, iter_range=(0, (mean_image_dim/10)-10), threshold=0.5, batch_size=1, maximum_blob_size=100000000)
    mask_sizes = mask_sizes + current_mask_sizes
    
    # mask has (1,1,image_height, image_width), modify to (image_height, image_width, 1), then stack three times to (image_height, image_width, 3)
    mask = mask.squeeze()
    mask = transforms.functional.to_pil_image(mask)
#    mask = mask[:,:,None]
#    mask = np.concatenate([mask,mask,mask], axis=2)
#    mask = mask.astype(np.bool)
#    
    # open random other image to take lesion from
    other_image_name = np.random.choice(list_of_test_images)
    while other_image_name == image_name:
        other_image_name = np.random.choice(list_of_test_images)
    other_image = Image.open(os.path.join(image_base_path,"test",other_image_name))# this is the image to take the anomalies from
#    other_image = np.array(other_image)
    other_image = other_image.resize(image.size) # needs to be resized so that mask fits
    
    # paste part of the "other_image" onto the original image
    modified_image = image.copy()
    modified_image.paste(other_image, mask=mask)
    
#    modified_image = Image.fromarray(modified_image)
#    mask = Image.fromarray(mask[:,:,0]) # just one dimension of the mask is enough. the three channels are the same, anyway
    
    # save every other image to val set, every other image to test set
    if count % 2 == 0:
        modified_image.save(os.path.join(pathol_image_base_path, "val", "images", image_name))
        mask.save(os.path.join(pathol_image_base_path, "val", "label_images", image_name))
    else:
        modified_image.save(os.path.join(pathol_image_base_path, "test", "images", image_name))
        mask.save(os.path.join(pathol_image_base_path, "test", "label_images", image_name))

# Calculate and print some mask size stats
mask_size_0 = np.empty(len(mask_sizes))
mask_size_1 = np.empty(len(mask_sizes))

for idx, size in enumerate(mask_sizes):
    mask_size_0[idx] = size[0]
    mask_size_1[idx] = size[1]
    
np.savetxt("mask_size_0.txt", mask_size_0)
np.savetxt("mask_size_1.txt", mask_size_1)

print("Min: ", np.min(mask_size_0), "median: ", np.median(mask_size_0), "max: ", np.max(mask_size_0), "sd: ", np.std(mask_size_0))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(mask_size_0, bins= 50)
ax.set_xlabel("Spatial extent of lesion in image dimension 0 [pixels]")
ax.set_ylabel("Number of lesions")


print("Min: ", np.min(mask_size_1), "median: ", np.median(mask_size_1), "max: ", np.max(mask_size_1), "sd: ", np.std(mask_size_1))
plt.hist(mask_size_1)
# =============================================================================
# ### Visualise mask
# 
# batch_size = 8
# 

# 
# grid = make_grid(mask, nrow=batch_size, padding=10, normalize=False, range=None, scale_each=False, pad_value=0.5)
# show(grid)       
# =============================================================================

