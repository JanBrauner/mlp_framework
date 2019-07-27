# =============================================================================
# #%% Testing the MiasHealthy
# from data_providers import MiasHealthy
# import torch
# import torchvision.transforms as transforms
# import numpy as np
# import math
# 
# task = "regression"
# debug_mode=False 
# patch_size=(1024,1024)
# patch_location="central"
# mask_size=(64,64)
# 
# 
# transform = True
# gamma_factor=1
# rot_angle=30
# shear_angle=30
# translate_distance=(0.2,0.2)
# scale_factor=1.5
# 
# # =============================================================================
# # standard_transforms = [transforms.ToTensor(),
# #                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
# # =============================================================================
# 
# standard_transforms = [transforms.ToTensor()]
# 
# if transform:
#     augmentations = [transforms.RandomAffine(degrees=rot_angle, translate=translate_distance, 
#                                         scale=(1/scale_factor, scale_factor),
#                                         shear=shear_angle)]
#     transformer_train = transforms.Compose(augmentations + standard_transforms)
# else:
#     transformer_train = transforms.Compose(standard_transforms)
#     
# trainset = MiasHealthy(which_set="train", task=task,
#                  transformer=transformer_train,
#                  debug_mode=debug_mode, 
#                  patch_size=patch_size, patch_location=patch_location, mask_size=mask_size)
# 
# trainset_iter = iter(trainset)
# for iterations in range(np.random.randint(1,100)):
#     data = next(trainset_iter)
# 
# 
# def create_central_region_slice(image_size, size_central_region):
#     margins = ((image_size[1]-size_central_region[0])/2, 
#                (image_size[2]-size_central_region[1])/2) # size of margins in dimensions 1 and 2 (relative to the 3-D tensor) between the image borders and the patch borders
#     
#     central_region_slice = np.s_[:, 
#                       math.ceil(margins[0]):math.ceil(image_size[1]-margins[0]), 
#                       math.ceil(margins[1]):math.ceil(image_size[2]-margins[1])]
#     return central_region_slice
# 
# # tests    
# assert type(data[0]) is torch.Tensor
# assert type(data[1]) is torch.Tensor
# assert data[0].size() == (1,) + patch_size
# assert data[1].size() == (1,) + mask_size
# 
# central_region_slice = create_central_region_slice(data[0].size(), mask_size)
# assert torch.all(torch.eq(data[0][central_region_slice], 0))# torch.all(torch.eq(data[0][central_region_slice], data[1]))
# 
# # visualisation
# composed = data[0].clone()
# composed[central_region_slice] = data[1]
# simple_back_transformer = transforms.ToPILImage()
# image = simple_back_transformer(data[0])
# target = simple_back_transformer(data[1])
# composed = simple_back_transformer(composed)
# image.show()
# target.show()
# composed.show()
# 
# =============================================================================

# =============================================================================
# #%% Testing DatasetWithAnomlies
# 
# ### prepare
# import torch
# import torchvision.transforms as transforms
# import torchvision
# import numpy as np
# import math
# from data_providers import DatasetWithAnomalies
# import matplotlib.pyplot as plt
# import os
# from PIL import Image
# 
# def create_central_region_slice(image_size, size_central_region):
#     # create slice of the central region of an image (dimensions (CxHxW)), when the size of the central region is central_region_size (HxW)
#     margins = ((image_size[2]-size_central_region[0])/2, 
#                (image_size[3]-size_central_region[1])/2) # size of margins in dimensions 1 and 2 (relative to the 3-D tensor) between the image borders and the patch borders
#     
#     central_region_slice = np.s_[:,:, 
#                       math.ceil(margins[0]):math.ceil(image_size[2]-margins[0]), 
#                       math.ceil(margins[1]):math.ceil(image_size[3]-margins[1])]
#     return central_region_slice
# 
# def show(img, cax):
#     # show PIL image
#     npimg = img.numpy()
#     cax.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
# 
# 
# # paths:
# image_path = "C:\\Users\\MC JB\\msc_project\\mlp_framework\\data\\MiasPathological\\test\\images"
# 
# which_set = "test"
# 
# #transformations = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
# debug_mode = True
# patch_size = (128,128)
# mask_size = (64,64)
# patch_stride = (10,10)
# batch_size = 5
# 
# 
# transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# 
# data = DatasetWithAnomalies(which_set, transformer, debug_mode, patch_size=patch_size, patch_stride=patch_stride, mask_size=mask_size)
# sample = data[1]
# image_batch_idx = 10
# 
# ### Basic tests:
# assert sample[0].shape == (1,) + patch_size
# assert sample[1].shape == (1,) + mask_size
# assert type(sample[2]) is str
# 
# 
# ### data_loader
# kwargs_dataloader = {"batch_size": batch_size,
#                      "num_workers": 0}
# data_loader = torch.utils.data.DataLoader(data, shuffle=False, **kwargs_dataloader)
# 
# # load n-th batch, n = image_batch_idx
# cnt = 0
# for inputs, targets, image_names, slices in data_loader:
#     cnt += 1
#     if cnt >= image_batch_idx:
#         break
# 
# # reload the respective original images, to see if the slide positions are correct
# for idx, image_name in enumerate(image_names):
#     reloaded_ori_full_image = Image.open(os.path.join(image_path, image_name))
#     reloaded_ori_full_image = transformer(reloaded_ori_full_image)
#     current_slice = np.s_[:,
#                           np.s_[slices["1_start"][idx]:slices["1_stop"][idx]],
#                           np.s_[slices["2_start"][idx]:slices["2_stop"][idx]],]
#     reloaded_ori_image = reloaded_ori_full_image[current_slice].clone().detach()
#     reloaded_ori_image = torch.unsqueeze(reloaded_ori_image,0)
#     if idx == 0:
#         reloaded_ori_images = reloaded_ori_image
#     else:
#         reloaded_ori_images = torch.cat((reloaded_ori_images,reloaded_ori_image), 0) 
#     
# 
# 
# ##### test if the outputs fit together:
# # create original images by combining inputs (masked out) and targets
# central_region_slice = create_central_region_slice(inputs.shape, mask_size)
# original_images = inputs.clone().detach()
# original_images[central_region_slice] = targets
# 
# 
# ### inverse normalization of all images
# inv_normalize = transforms.Normalize((-0.5/0.5,), (1/0.5,))
# 
# for images in [inputs, original_images, reloaded_ori_images, targets]:
#     for idx,image in enumerate(images): # since these are a batch of images, but transforms work on indivdual images
#         images[idx,:,:,:] = inv_normalize(image)
# 
# 
# ### THE ONE ASSERTION: If that works, the whole thing probably works
# assert torch.all(torch.eq(targets, reloaded_ori_images))
# 
# 
# 
# ###### Display images
# ### create image grid
# grid_parameters = {
#         "nrow":batch_size, 
#         "padding":10, 
#         "normalize":False, 
#         "range":None, 
#         "scale_each":False, 
#         "pad_value":0}
# 
# originals_grid = torchvision.utils.make_grid(original_images, **grid_parameters)
# inputs_grid = torchvision.utils.make_grid(inputs, **grid_parameters)
# targets_grid = torchvision.utils.make_grid(targets, **grid_parameters)
# reloaded_ori_grid = torchvision.utils.make_grid(reloaded_ori_images, **grid_parameters)
# 
# ### plot images: 
# """
# first row: inputs
# second row: inputs + targets combined, should match with first row
# third row: targets
# fourth row: regions of the reloaded original image, retrieved with slice passed from data provider
# -> third and fourth row should be the same
# two rows show input
# """
# 
# fig = plt.figure()
# cax = fig.add_subplot(411)
# show(inputs_grid, cax)
# 
# cax = fig.add_subplot(412)
# show(originals_grid, cax)
# 
# 
# cax = fig.add_subplot(413)
# show(reloaded_ori_grid, cax)
# 
# cax = fig.add_subplot(414)
# show(reloaded_ori_grid, cax)
# =============================================================================




# =============================================================================
# #%% counting anomaly maps
# anomaly_detection_base_dir = os.path.abspath(os.path.join("results", "anomaly_detection"))
# experiment_names = ['CE_DTD_r2_stand_scale_1___AD_window_mean',
# 'CE_DTD_r2_stand_scale_1___AD_window_min',
# 'CE_DTD_r2_stand_scale_1___AD_window_max',
# 'CE_DTD_r2_stand_scale_0p5___AD_window_mean',
# 'CE_DTD_r2_stand_scale_0p5___AD_window_min',
# 'CE_DTD_r2_stand_scale_0p5___AD_window_max',
# 'CE_DTD_r2_stand_small_mask___AD_window_mean',
# 'CE_DTD_r2_stand_small_mask___AD_window_min',
# 'CE_DTD_r2_stand_small_mask___AD_window_max',
# 'CE_DTD_r2_stand_large_context___AD_window_mean',
# 'CE_DTD_r2_stand_large_context___AD_window_min',
# 'CE_DTD_r2_stand_large_context___AD_window_max',
# 'CE_DTD_r2_prob_scale_1___AD_window_mean',
# 'CE_DTD_r2_prob_scale_1___AD_window_min',
# 'CE_DTD_r2_prob_scale_1___AD_window_max',
# 'CE_DTD_r2_prob_scale_0p5___AD_window_mean',
# 'CE_DTD_r2_prob_scale_0p5___AD_window_min',
# 'CE_DTD_r2_prob_scale_0p5___AD_window_max',
# 'CE_DTD_r2_prob_small_mask___AD_window_mean',
# 'CE_DTD_r2_prob_small_mask___AD_window_min',
# 'CE_DTD_r2_prob_small_mask___AD_window_max',
# 'CE_DTD_r2_prob_large_context___AD_window_mean',
# 'CE_DTD_r2_prob_large_context___AD_window_min',
# 'CE_DTD_r2_prob_large_context___AD_window_max']
# 
# for experiment_name in experiment_names:
#     for which_set in ["val", "test"]:
#         anomaly_maps_dir = os.path.join(anomaly_detection_base_dir, experiment_name, "anomaly_maps", which_set)
#         try:
#             num_anomaly_maps = len(os.listdir(anomaly_maps_dir))
#             print("{}, {} set: {}".format(experiment_name, which_set, str(num_anomaly_maps)))
#         except:
#             pass
# =============================================================================
        
#%% working space
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
s
npimg = np.arange(9).reshape((3,3)) + 0.5
image_from_array = Image.fromarray(npimg)

tensor = torch.from_numpy(npimg).type(torch.float)
image_from_tensor = transforms.functional.to_pil_image(tensor)

print("Image from array: ", np.array(image_from_array))
print("Image from tensor: ", np.array(image_from_tensor))

array = np.arange(9).reshape((3,3)) + 0.5

tensor = torch.from_numpy(tensor).type(torch.float)

image = transforms.functional.to_pil_image(tensor, mode="F")
image = np.array(image)


print("Tensor: ", tensor)
print("Image: ", image)
