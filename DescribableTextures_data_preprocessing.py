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
list_of_test_images = os.listdir(os.path.join(image_base_path, "test"))

# delete all files in resepctive folders:
for which_set in ["test"]:
    for image_type in ["images","label_images"]:
        for file_name in os.listdir(os.path.join(pathol_image_base_path, which_set, image_type)):
            os.remove(os.path.join(pathol_image_base_path, which_set, image_type, file_name))


# modify each test image from test folder of the normal DT dataset and copy the modfied version to the new folder. Also save mask.
count = 0
for image_name in list_of_test_images:
    count += 1
    image = Image.open(os.path.join(image_base_path,"test",image_name))
    npimage = np.array(image) # 
    mean_image_dim = (npimage.shape[0]+npimage.shape[1])/2
    mask = mask_utils.batch_multi_random_blobs(img_size=(npimage.shape[2], npimage.shape[0], npimage.shape[1]), max_num_blobs=2, iter_range=(0, (mean_image_dim/10)-10), threshold=0.5, batch_size=1, maximum_blob_size=100000000)
    
    # mask has (1,1,image_height, image_width), modify to (image_height, image_width, 1), then stack three times to (image_height, image_width, 3)
    mask = mask.squeeze()
    mask = transforms.functional.to_pil_image(mask)
#    mask = mask[:,:,None]
#    mask = np.concatenate([mask,mask,mask], axis=2)
#    mask = mask.astype(np.bool)
#    
    
    other_image_name = np.random.choice(list_of_test_images)
    while other_image_name == image_name:
        other_image_name = np.random.choice(list_of_test_images)
    other_image = Image.open(os.path.join(image_base_path,"test",other_image_name))# this is the image to take the anomalies from
#    other_image = np.array(other_image)
    other_image = other_image.resize(image.size)
    
    modified_image = image.copy()
    modified_image.paste(other_image, mask=mask)
    
#    modified_image = Image.fromarray(modified_image)
#    mask = Image.fromarray(mask[:,:,0]) # just one dimension of the mask is enough. the three channels are the same, anyway
    
    
    modified_image.save(os.path.join(pathol_image_base_path, "test", "images", image_name))
    mask.save(os.path.join(pathol_image_base_path, "test", "label_images", image_name))

# =============================================================================
# ### Visualise mask
# 
# batch_size = 8
# 

# 
# grid = make_grid(mask, nrow=batch_size, padding=10, normalize=False, range=None, scale_each=False, pad_value=0.5)
# show(grid)       
# =============================================================================

