"""
- Combine anomaly maps from several models/ patch sizes/ mask sizes/ window aggregation methods/...
- determine AUC values and other scores for the new maps
"""

import os, sys
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import itertools

from arg_extractor import get_args
from misc_utils import get_aucroc
from storage_utils import save_statistics
from visualisation_utils import show

#%% single example

# parameters
experiment_names = ["CE_DTD_r2_prob_scale_1___AD_window_mean"]
save_name = "CE_DTD_r2_prob_scale_1___AD_window_mean_save_test"
which_set = "both"

combination_method = "mean"
AD_margins = [128,128]

# paths
anomaly_detection_base_dir = os.path.abspath(os.path.join("results", "anomaly_detection"))
label_image_base_dir = os.path.abspath(os.path.join("data")) 
save_base_dir = anomaly_detection_base_dir


#%%
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def load_all_anomaly_maps(anomaly_map_dirs, image_name, transformer):
    """ 
    Load all anomaly_maps for a given image from several anomaly map directories
    
    outputs Tensor with anomaly maps: (nxCxHxW)
    """
    for anomaly_map_dir in anomaly_map_dirs:
        anomaly_map_path = os.path.join(anomaly_map_dir, image_name)
        anomaly_map = torch.load(anomaly_map_path)
        
        # normalise anomaly map (otherwise aggregating doesn't really make sense)
        anomaly_map = anomaly_map/anomaly_map.max()
        try: # if anomaly_maps already exists
            anomaly_maps = torch.cat((anomaly_maps, anomaly_map.unsqueeze(0)), dim=0) # continue here. That probably fixed many things. Write down learning here.
        except UnboundLocalError:
            anomaly_maps = anomaly_map.unsqueeze(0)
    return anomaly_maps

#
def combine_anomaly_maps(experiment_names,
                         save_name,
                         which_set,
                         combination_method, 
                         AD_margins, 
                         anomaly_detection_base_dir=anomaly_detection_base_dir,
                         label_image_base_dir=label_image_base_dir,
                         save_base_dir=save_base_dir):
    
    if which_set == "both":
        combine_anomaly_maps(experiment_names, save_name, "val", combination_method, AD_margins, anomaly_detection_base_dir, label_image_base_dir, save_base_dir)
        combine_anomaly_maps(experiment_names, save_name, "test", combination_method, AD_margins, anomaly_detection_base_dir, label_image_base_dir, save_base_dir)
        return
    
    # Initialize
    transformer = transforms.ToTensor() # there are certainly other ways of doing this, but this is consistent with AnomalyDetectionExperiment and DatasetWithAnomalies
    anomaly_map_dirs = []
    stats_dict = {"aucroc":[]} # a dict that keeps the measures of agreement between pixel-wise anomaly score and ground-truth labels, for each image. Current,y AUC is the only measure.

    
        
    for experiment_name in experiment_names:
        anomaly_map_dirs.append(os.path.join(anomaly_detection_base_dir, experiment_name, "anomaly_maps", which_set))
        
    image_names = os.listdir(anomaly_map_dirs[0]) # all anomaly map dirs should contain anomaly maps of the same images. Also, I load from this folder, instead of the dataset folder, in case anomaly maps were only created for a part of the available images
    
    for image_idx, image_name in enumerate(image_names):
        
        # load anomaly maps
        anomaly_maps = load_all_anomaly_maps(anomaly_map_dirs, image_name, transformer)
        
        # combine anomaly maps
        if combination_method == "min":
            anomaly_map, _ = torch.min(anomaly_maps, dim=0)
        elif combination_method == "mean":
            anomaly_map = torch.mean(anomaly_maps, dim=0)
        elif combination_method == "max":
            anomaly_map, _ = torch.max(anomaly_maps, dim=0)
                    
        
        # load label image
        with HiddenPrints():
            args, _ = get_args(experiment_name)
        anomaly_dataset_name = args.anomaly_dataset_name
        label_image_path  = os.path.join(label_image_base_dir, 
                                         anomaly_dataset_name, 
                                         which_set,
                                         "label_images",
                                         image_name)   
        label_image = Image.open(label_image_path)
        label_image = transformer(label_image)
        
        
        # cut out region for calculation of AUC and other scores
        if AD_margins is not None:
            slice_considered_for_AD = np.s_[:,
                                            AD_margins[0]:anomaly_map.shape[1]-AD_margins[0],
                                            AD_margins[1]:anomaly_map.shape[2]-AD_margins[1]]
            anomaly_map = anomaly_map[slice_considered_for_AD]
            label_image = label_image[slice_considered_for_AD]
        
# =============================================================================
#         # maybe later on this will be the better choice. Right onw, with only aucroc, it's not needed
#         calculate_agreement_between_anomaly_score_and_labels(anomaly_map, label_image)
# 
# =============================================================================
        
        #calculate stats
        stats_dict["aucroc"].append(get_aucroc(label_image, anomaly_map))
        
        # save stats
        save_path = os.path.join(save_base_dir, save_name, "tables")
        try:
            os.makedirs(save_path)
        except FileExistsError:
            pass
        save_statistics(experiment_log_dir=save_path, filename=which_set +'_summary.csv',
                            stats_dict=stats_dict, current_epoch=image_idx, continue_from_mode=True if image_idx>0 else False, save_full_dict=False) # save statistics to stats file.
 
    # print mean results:
    print("{} set results:".format(which_set))
    for key, list_of_values in stats_dict.items():
        list_of_non_nan_values = [x for x in list_of_values if not np.isnan(x)]
        mean_value = sum(list_of_non_nan_values)/len(list_of_non_nan_values)
        print("Mean ", key, ": ", "{:.4f}".format(mean_value)) 
            
    return

# =============================================================================
# #%% single example
# 
# anomaly_maps = combine_anomaly_maps(experiment_names=experiment_names,
#                                      save_name=save_name,
#                                      which_set=which_set,
#                                      combination_method=combination_method, 
#                                      AD_margins=AD_margins, 
#                                      anomaly_detection_base_dir=anomaly_detection_base_dir,
#                                      label_image_base_dir=label_image_base_dir,
#                                      save_base_dir=save_base_dir)
# 
# =============================================================================

#%% many examples
which_set = "both"
AD_margins = [128,128]



exp_name_string = "CE_DTD_r2_{model_type}_{model_setting}___AD_window_{window_aggregation_method}"
save_string = "CE_DTD_r2_{model_type}___ADcomb_{setting_0}{setting_1}{setting_2}{setting_3}_win_{window_aggregation_method}_comb_{model_combination_method}"

model_settings = ["Sc1", "Sc05", "Sm", "Lc"] # scale 1, scale 0.5, small mask, large context

all_combinations = []
for r in range(2,len(model_settings)+1):
    all_combinations += itertools.combinations(model_settings, r)

model_types = ["stand","prob"]
window_aggregation_methods = ["mean", "max"]
model_combination_methods = ["min", "mean", "max"]

translator = {"Sc1": "scale_1",
              "Sc05": "scale_0p5", 
              "Sm": "small_mask", 
              "Lc": "large_context"}

iterator = itertools.product(model_types, all_combinations, window_aggregation_methods, model_combination_methods)
save_names = []
for model_type, model_setting_list, window_aggregation_method, model_combination_method in iterator:
    experiment_names = []
    for model_setting in model_setting_list:
        experiment_names.append(exp_name_string.format(model_type=model_type, 
                                                       model_setting=translator[model_setting],
                                                       window_aggregation_method=window_aggregation_method))
    save_name = save_string.format(model_type=model_type, 
                                   setting_0=model_setting_list[0],
                                   setting_1=model_setting_list[1],
                                   setting_2=(model_setting_list + ("",""))[2], # just extend list, so that in case the list is only 2 long, empty strings get sampled
                                   setting_3=(model_setting_list + ("",""))[3],
                                   window_aggregation_method=window_aggregation_method,
                                   model_combination_method=model_combination_method)
    
    save_names.append(save_name)
    
    combine_anomaly_maps(experiment_names=experiment_names,
                         save_name=save_name,
                         which_set=which_set,
                         combination_method=model_combination_method, 
                         AD_margins=AD_margins, 
                         anomaly_detection_base_dir=anomaly_detection_base_dir,
                         label_image_base_dir=label_image_base_dir,
                         save_base_dir=save_base_dir)

with open("r2___ADcomb_exp_names.txt", "w") as f:
    for name in save_names:
        f.writelines("'{}',\n".format(name))

#for stand_or_prob in ["stand","prob"]:
#    for model_combination_method in ["min", "mean", "max"]:
#        for model_setting_list in all_combinations:
#            
#
#
#
#
#        
#
#        call anomaly_maps multiple times in a loop