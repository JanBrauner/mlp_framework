"""
- Combine anomaly maps from several models/ patch sizes/ mask sizes/ window aggregation methods/...
- determine AUC values and other scores for the new maps
"""

import os
import PIL as Image
import torchvision.transforms as transforms

# parameters
combination_method = "mean"
experiment_names = []

# paths
anomaly_map_base_dir = os.path.abspath(os.path.join("results", "anomaly_detection")


#
transformer = transforms.ToTensor() # there are certainly other ways of doing this, but this is consistent with AnomalyDetectionExperiment and DatasetWithAnomalies

anomaly_map_dirs = []
for experiment_name in experiment_names:
    anomaly_map_dirs.append(os.path.join(anomaly_map_base_dir, experiment_name, "anomaly_maps"))
    
    # which set needs to appear somewhere!!!!
    
    
    # something here clearly should be a function....
    image_names = os.listdir(anomaly_map_dirs[0]) # all anomaly map dirs should contain anomaly maps of the same images. Also, I load from this folder, instead of the dataset folder, in case anomaly maps were only created for a part of the available images
    for image_name in image_names:
        for anomaly_map_dir in anomaly_map_dirs:
            anomaly_map_path = os.path.join(anomaly_map_dir, image_name)
            anomaly_map = Image.open(anomaly_map_path)
            anomaly_map = transformer(anomaly_map)
            try: # if anomaly_maps already exists
                anomaly_maps = torch.cat(anomaly_maps, anomaly_map, dim=0)
            except:
                anomaly_maps = anomaly_map.unsqueeze(0)
            
    
    for anomaly_map_dir in anomaly_map_dirs: