"""
- Combine anomaly maps from several models/ patch sizes/ mask sizes/ window aggregation methods/...
- determine AUC values and other scores for the new maps
"""

import os, sys
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

from arg_extractor import get_args
from misc_utils import get_aucroc
from storage_utils import save_statistics
from visualisation_utils import show
# =============================================================================
# 
# def calculate_agreement_between_anomaly_score_and_labels(anomaly_map, label_image):
#     ### calculate measures of agreement 
#     # AUC: currently the only measure of agreement
#     if self.measure_of_anomaly == "absolute distance" or self.measure_of_anomaly == "likelihood": #then all anomly scores will be in [0,infinity], and higher scores will mean more anomaly, so no further preprocessing is needed to calculate AUC:
#         aucroc = get_aucroc(label_image, anomaly_map)            
#     
#     self.stats_dict["aucroc"].append(aucroc)
# =============================================================================
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
        try: # if anomaly_maps already exists
            anomaly_maps = torch.cat(anomaly_maps, anomaly_map.unsqueeze(0), dim=0)
        except:
            anomaly_maps = anomaly_map.unsqueeze(0)
    return anomaly_maps


#%%

# parameters
experiment_names = ["CE_DTD_r2_prob_scale_1___AD_window_mean"]
save_name = "CE_DTD_r2_prob_scale_1___AD_window_mean"
which_set = "val"

combination_method = "mean"
AD_margins = [128,128]

# paths
anomaly_detection_base_dir = os.path.abspath(os.path.join("results", "anomaly_detection"))
label_image_base_dir = os.path.abspath(os.path.join("data")) 
save_base_dir = anomaly_detection_base_dir

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
        return anomaly_maps
        
        # combine anomaly maps
        if combination_method == "min":
            anomaly_map = torch.min(anomaly_maps, dim=0)
        if combination_method == "mean":
            anomaly_map = torch.mean(anomaly_maps, dim=0)
        if combination_method == "max":
            anomaly_map = torch.max(anomaly_maps, dim=0)
                    
        
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
        save_statistics(experiment_log_dir=save_path, filename=which_set +'_summary.csv',
                            stats_dict=stats_dict, current_epoch=image_idx, continue_from_mode=True if image_idx>0 else False, save_full_dict=False) # save statistics to stats file.
 
    # print mean results:
    print("{} set results:".format(which_set))
    for key, list_of_values in stats_dict.items():
        list_of_non_nan_values = [x for x in list_of_values if not np.isnan(x)]
        mean_value = sum(list_of_non_nan_values)/len(list_of_non_nan_values)
        print("Mean ", key, ": ", "{:.4f}".format(mean_value)) 
            
    return

anomaly_maps = combine_anomaly_maps(experiment_names=experiment_names,
                     save_name=save_name,
                     which_set=which_set,
                     combination_method=combination_method, 
                     AD_margins=AD_margins, 
                     anomaly_detection_base_dir=anomaly_detection_base_dir,
                     label_image_base_dir=label_image_base_dir,
                     save_base_dir=save_base_dir)