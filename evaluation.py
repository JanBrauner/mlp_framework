"""
ToDo:

    First thing: debug and make sure this shit works.
    
    Then: integrate with pipeline. First thing: just copy-paste, or really include with experiment builder?
    I need to include this in a a pipeline so that it acutally works, e.g. with loading a model:
        Do I want this to be part of experiment builder, of alone standing?
        I definitely want it to work without having to train a new model every time!
        
    If I have to do stuff on the cluster, I might have to send stuff to the GPU, detach at the right times, ...
    
    At the end, refactor DataSet nicely
    enable combining heatmaps from several models?

    I could use a more sophisticated tracking system of multiple stats, like the one in ExperimentBuilder
    
    Show evaluation process
    
    Do I want to save anomaly maps? Prolly yes. Then I need to implement this

    do some testing
        DataSet works
        normalisation_map works
        Let's just first see if it does what I think it does. If it does, then no more testing is necessary
    
    Do I need multiple folders for saving the maps?
    
    Script generator: simply have an option to do trainig, anomaly detection, or both
    
    Pipeline/script generation: probably a good solution would be to have anomaly_detection_experimemt_names
        Think about this for a second, how this would be nice to analyse as well!
        
    I need to save the AUC scores somewhere
    Maybe include further scores
            
    Script generator: include all the new args and update the description of the args that have double meaning
At the end:
    
    
Notes:
    - write everything to be [BxCxHxW]-compatible. I might want to do it on the cluster.
"""



### prepare
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision
import numpy as np
import math
from data_providers import DescribableTexturesPathological
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

import data_providers as data_providers
import model_architectures
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
from misc_utils import get_aucroc
from storage_utils import load_best_model_state_dict


def create_central_region_slice(image_size, size_central_region):
    # create slice of the central region of an image (dimensions (CxHxW)), when the size of the central region is central_region_size (HxW)
    margins = ((image_size[2]-size_central_region[0])/2, 
               (image_size[3]-size_central_region[1])/2) # size of margins in dimensions 1 and 2 (relative to the 3-D tensor) between the image borders and the patch borders
    
    central_region_slice = np.s_[:,:, 
                      math.ceil(margins[0]):math.ceil(image_size[2]-margins[0]), 
                      math.ceil(margins[1]):math.ceil(image_size[3]-margins[1])]
    return central_region_slice


def normalise_anomaly_map(anomaly_map, normalisation_map, window_aggregation_method):
    if window_aggregation_method == "mean": # how we normalise the anomaly_map might depend on the window aggregation method
    # normalise anomaly score maps
        normalisation_map[normalisation_map == 0] = 1 # change zeros in the normalisation factor to 1
        anomaly_map = anomaly_map / normalisation_map
    return anomaly_map

def calculate_agreement_between_anomaly_score_and_labels(image_idx, anomaly_map, normalisation_map, measure_of_anomaly, window_aggregation_method):

    # load ground truth segmentation label image
    label_image = data.get_label_image(image_idx) # For example, when current_image_idx just jumped from 3 to 4, that means that image "3" is finished.
    
    # calculate measures of agreement (currently: AUC)
    if measure_of_anomaly == "absolute distance": #then all anomly scores will be in [0,1], so no further preprocessing is needed to calculate AUC:
        aucroc = get_aucroc(label_image, anomaly_map)
    
    return aucroc


args, device = get_args()  # get arguments from command line/json config.

# set random seeds
rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(seed=args.seed)

# create dataset:
data_loader, image_list, image_sizes = data_providers.create_dataset_with_anomalies(
        anomaly_dataset_name=args.anomaly_dataset_name, which_set=args.which_set, 
        normalisation=args.normalisation, batch_size=args.batch_size, 
        patch_size=args.patch_size, patch_stride=args.patch_stride, mask_size=args.mask_size, 
        num_workers=args.num_workers, debug_mode=args.debug_mode)



# create model
model = model_architectures.create_model(args)

# Load state dict from  best epoch of that experiment
model_dir = os.path.join("results", experiment_name, "saved_models")
state_dict = load_best_model_state_dict(model_dir=model_dir, is_gpu=False) # this flag probably shouldn't be called "is_gpu", since it really rather is about moving from GPU to CPU

model.load_state_dict(state_dict=state_dict["network"])

#### Probably here the class begins!
# build experiment

# run experiment


# paths:



class AnomalyDetectionExperiment(object): # continue here
    def __init__(self, args):
        anomaly_map_dir = os.path.join("results", "anomaly_detection", self.experiment_name + "-" + self.anomaly_detection_experiment_name)
        try:
            os.mkdir(anomaly_map_dir)
        except FileExistsError:
            pass
which_set = "test"

#transformations = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
debug_mode = True
patch_size = (128,128)
mask_size = (64,64)
patch_stride = (30,30)
batch_size = 5
measure_of_anomaly = "absolute distance"
window_aggregation_method = "mean"
save_anomaly_maps = True


experiment_name = "CE_DTD_random_patch_test_1"
args, device = get_args(experiment_name)  # get arguments from command line. Run local debugging with settings as specified in CE_cpu_dev
# args, device = get_args()  # get arguments from command line. Run local debugging with settings as specified in CE_cpu_dev
#args, device = get_args("CE_test") # for local debugging

args.num_image_channels = 3 ### obviously delete this later on!
args.use_gpu = False
args.num_workers = 0



model.eval()






#transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#
#
#
#
##%% HERE IS WHERE THE INTERESTING STUFF BEGINS
#data = DescribableTexturesPathological(which_set, transformer, debug_mode=debug_mode, patch_size=patch_size, patch_stride=patch_stride, mask_size=mask_size)
#image_list = data.image_list
#image_sizes = data.image_sizes
#
#
#### data_loader
#kwargs_dataloader = {"batch_size": batch_size,
#                     "num_workers": 0}
#data_loader = torch.utils.data.DataLoader(data, shuffle=False, **kwargs_dataloader)

model.eval()
num_finished_images = -1 # the data loader works through the test set images in order. num_finished_images is a counter that ticks up everytime one image is finished
aucroc_per_image = np.empty(len(image_list)) # collects the AUC-ROC per image

for inputs, targets, image_idxs, slices in data_loader:
    outputs = model.forward(inputs)
    if measure_of_anomaly == "absolute distance":
        anomaly_score = torch.abs(outputs - targets) # pixelwise anomaly score, for each of the images in the batch
        anomaly_score = torch.mean(anomaly_score, dim=1, keepdim=True) # take the mean over the channels
    
    # the following for-loop deals with translating the pixelwise anomaly for one sliding window position (and thus a score relative to an image patch) into an anomaly score for the full image
    for batch_idx in range(len(image_idxs)): # for each image in the batch. "in range(batch_size)" leads to error, because the last batch is smaller that the batch size
        # Get the index of the full image that the current patch was taken from. The index is relative to image_list and image_sizes
        current_image_idx = int(image_idxs[batch_idx].detach().numpy())
        assert num_finished_images <= current_image_idx, "This assertion fails if the dataloader does not strucutre the batches so that the order of images/patches WITHIN the batch does still correspond to image_list" # Basically, I am sure that __getitem__() gets items in the right order, but I am unsure if the order gets imxed up within the minibatch by the DataLoader. Probably best to leave that assertion in, since this will throw a bug if the behaviour of DataLoader is changed in future PyTorch versions.

        
        if current_image_idx > num_finished_images: # Upon starting the with the first patch, or whenever we have moved on to the next image
            num_finished_images += 1
            if num_finished_images > 0: # Whenever we have moved to the next image, calculate agreement between our anomaly score and the ground truth segmentation. (Obviously don't do this when we are jstus tarting with the first patch)
                anomaly_map = normalise_anomaly_map(anomaly_map,normalisation_map,window_aggregation_method)
                aucroc_per_image[current_image_idx-1] = calculate_agreement_between_anomaly_score_and_labels(
                        image_idx=current_image_idx-1, anomaly_map=anomaly_map, 
                        normalisation_map=normalisation_map, measure_of_anomaly=measure_of_anomaly, 
                        window_aggregation_method=window_aggregation_method)
                
                if save_anomaly_maps:
                    anomaly_map = F.to_pil_image(anomaly_map)
                    anomaly_map.save(os.path.join(anomaly_map_dir,image_list[current_image_idx -1]))
                
            # Upon starting the with the first patch, or whenever we have moved on to the next image, create new anomaly maps and normalisation maps
            current_image_height = image_sizes[current_image_idx][1]
            current_image_width = image_sizes[current_image_idx][2]
            
            anomaly_map = torch.zeros((1,current_image_height, current_image_width)) # anomaly score heat maps for every image. Initialise as constant zero tensor of the same size as the full image
            normalisation_map = torch.zeros((1,current_image_height, current_image_width)) # for every image, keep score of how often a given pixel has appeared in a sliding window, for calculation of average scores. Initialise as constant zero tensor of the same size as the full image
        
        # Now the part that happens for every image-patch(!): update the relevant part of the current anomaly_score map:
        current_slice = np.s_[:,
                              np.s_[slices["1_start"][batch_idx]:slices["1_stop"][batch_idx]],
                              np.s_[slices["2_start"][batch_idx]:slices["2_stop"][batch_idx]]]

        if window_aggregation_method == "mean":
            anomaly_map[current_slice] += anomaly_score[batch_idx,:,:,:]
            normalisation_map[current_slice] += 1


# also calculate results for the last image ()
aucroc_per_image[current_image_idx] = calculate_agreement_between_anomaly_score_and_labels(
        image_idx=current_image_idx, anomaly_map=anomaly_map, 
        normalisation_map=normalisation_map, measure_of_anomaly=measure_of_anomaly, 
        window_aggregation_method=window_aggregation_method)

if save_anomaly_maps:
    anomaly_map = F.to_pil_image(anomaly_map)
    anomaly_map.save(os.path.join(anomaly_map_dir,image_list[current_image_idx]))
                


aucroc_mn = np.mean(aucroc_per_image)




    
    

# =============================================================================
# 
# ### normalise anomaly score  maps
# for image_idx in range(len(image_list)):
#     normalisation_maps[image_idx][normalisation_maps[image_idx] == 0] = 1 # change zeros in the normalisation factor to 1
#     anomaly_maps[image_idx] = anomaly_maps[image_idx] / normalisation_maps[image_idx]
#     
# 
# 
# # =============================================================================
# # 
# # ### combine anomaly scores - REPLACED BECAUSE OF MEMORY ISSUES
# # all_anomaly_scores = {}
# # for image_name, image_size in zip(image_list, image_sizes):
# #     if window_aggregation_method == "mean":
# #         combined_score_tensor = torch.zeros(image_size)
# #         windows_per_pixel = torch.zeros(image_size) # counts how many times a pixel appears in a window, for averaging
# #         for window_info in all_windows[image_name]:
# #             score_tensor = torch.zeros(image_size)
# #             score_tensor[window_info["slice relative to full image"]] = window_info["anomaly score"]
# #             combined_score_tensor = combined_score_tensor  + score_tensor
# #             windows_per_pixel[window_info["slice relative to full image"]] += 1
# #         windows_per_pixel[windows_per_pixel == 0] = 1
# #         combined_score_tensor = combined_score_tensor / windows_per_pixel
# #         all_anomaly_scores[image_name] = combined_score_tensor
# #         
# # =============================================================================
# ### testing
# show_idx = 1
# anomaly_score = anomaly_maps[show_idx].detach().numpy()
# anomaly_score = np.squeeze(anomaly_score)
# normalisation_map = normalisation_maps[show_idx].detach().numpy()
# normalisation_map = np.squeeze(normalisation_map)
# 
# plt.figure()
# plt.imshow(anomaly_score)
# plt.figure()
# plt.imshow(normalisation_map)
# 
# #%%
# ### Compare anomaly heat map with ground-truth labels:
# from sklearn.metrics import roc_auc_score
# def get_aucroc(y_true, output):
#     if torch.min(y_true.data) == 1 or torch.max(y_true.data) == 0:
#         aucroc = np.nan # return nan if there are only examples of one type in the batch, because AUCROC is not defined then. 
#     else:
#         y_true = y_true.cpu().detach().numpy().flatten()
#         output = output.cpu().detach().numpy().flatten()
#         aucroc = roc_auc_score(y_true,output)
#     return aucroc
# 
# 
# aucroc_per_image = np.empty(len(image_list))
# for idx, anomaly_map in enumerate(anomaly_maps):
#     label_image = data.get_label_image(idx)
#     
#     if measure_of_anomaly == "absolute distance": #then all anoamly scores will be in [0,1]:
#         aucroc_per_image[idx] = get_aucroc(label_image, anomaly_map)
#         
# aucroc_mn = np.mean(aucroc_per_image)
#         
#     
# =============================================================================
