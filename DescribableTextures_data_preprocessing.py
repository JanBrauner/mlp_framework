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
# parameters
test_images_per_category = 20

# 2/3rd of of images go to train set: copy images from train1.txt and test1.txt to the train subdirectory
#file_path = os.path.join(raw_data_path, "train1.txt")
#list_of_images = create_list_of_images(file_path)
list_of_images1 = create_list_of_images(os.path.join(raw_data_path, "train1.txt"))
list_of_images2 = create_list_of_images(os.path.join(raw_data_path, "test1.txt"))
list_of_images = list_of_images1 + list_of_images2
for image_name in list_of_images:
    copy2(os.path.join(raw_data_path, image_name), os.path.join(image_base_path, "train"))
    
# The remaining images are split between val and test: copy images from val1.txt to the validation subdirectory
list_of_images = create_list_of_images(os.path.join(raw_data_path, "val1.txt"))
for idx, image_name in enumerate(list_of_images):
    if idx % (40/test_images_per_category) == 0: # 40 imges of each category in that file
        copy2(os.path.join(raw_data_path, image_name), os.path.join(image_base_path, "test"))
    else:
        copy2(os.path.join(raw_data_path, image_name), os.path.join(image_base_path, "val"))

