### prepare
import torch
import numpy as np

import data_providers as data_providers
import model_architectures
from arg_extractor import get_args
from experiment_builder import AnomalyDetectionExperiment


# =============================================================================
# # load args
# args, device = get_args()  # get arguments from command line/json config.
# train_experiment_name = args.experiment_name.split("___")[0] # name of the experiment in which the model that we want to use for anomaly detection was trained
# anomaly_detection_experiment_name = args.experiment_name.split("___")[1] # name of the anomaly detection experiment
# 
# =============================================================================
### for debugging
args, device = get_args("CE_DTD_random_patch_test_1___AD_test")  # get arguments from command line/json config.
train_experiment_name = args.experiment_name.split("___")[0] # name of the experiment in which the model that we want to use for anomaly detection was trained
anomaly_detection_experiment_name = args.experiment_name.split("___")[1] # name of the anomaly detection experiment

args.use_gpu = False
args.num_workers = 0
args.debug_mode = True
args.AD_patch_stride = (100,100)

# set random seeds
rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(seed=args.seed)

# create dataset:
val_dataset, val_data_loader, test_dataset, test_data_loader = data_providers.create_dataset_with_anomalies(
                                                                anomaly_dataset_name=args.anomaly_dataset_name, 
                                                                normalisation=args.normalisation, 
                                                                batch_size=args.AD_batch_size, 
                                                                patch_size=args.patch_size, 
                                                                patch_stride=args.AD_patch_stride, 
                                                                mask_size=args.mask_size, 
                                                                num_workers=args.num_workers, 
                                                                debug_mode=args.debug_mode)


# create model
model = model_architectures.create_model(args)

# create experiment
experiment = AnomalyDetectionExperiment(experiment_name=train_experiment_name, 
                                        anomaly_detection_experiment_name=anomaly_detection_experiment_name,
                                        model=model, 
                                        device=device,
                                        val_data_loader=val_data_loader, 
                                        val_dataset=val_dataset,
                                        test_data_loader=test_data_loader, 
                                        test_dataset=test_dataset,
                                        measure_of_anomaly=args.measure_of_anomaly, 
                                        window_aggregation_method=args.window_aggregation_method, 
                                        save_anomaly_maps=args.save_anomaly_maps,
                                        use_gpu = args.use_gpu)

# run experiment
experiment.run_experiment()
