### prepare
import torch
import numpy as np
import math

import data_providers as data_providers
import model_architectures
from arg_extractor import get_args
from experiment_builder import AnomalyDetectionExperiment


# load args
args, device = get_args()  # get arguments from command line/json config.
train_experiment_name = args.experiment_name.split("___")[0] # name of the experiment in which the model that we want to use for anomaly detection was trained
anomaly_detection_experiment_name = args.experiment_name.split("___")[1] # name of the anomaly detection experiment


# set random seeds
rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(seed=args.seed)

# create dataset:
test_data_loader, test_dataset = data_providers.create_dataset_with_anomalies(
        anomaly_dataset_name=args.anomaly_dataset_name, which_set="test", 
        normalisation=args.normalisation, batch_size=args.AD_batch_size, 
        patch_size=args.patch_size, patch_stride=args.AD_patch_stride, mask_size=args.mask_size, 
        num_workers=args.num_workers, debug_mode=args.debug_mode)


# create model
model = model_architectures.create_model(args)

# create experiment
experiment = AnomalyDetectionExperiment(experiment_name=train_experiment_name, 
                                        anomaly_detection_experiment_name=anomaly_detection_experiment_name,
                                        model=model, 
                                        device=device,
                                        test_data_loader=test_data_loader, 
                                        test_dataset=test_dataset,
                                        measure_of_anomaly=args.measure_of_anomaly, 
                                        window_aggregation_method=args.window_aggregation_method, 
                                        save_anomaly_maps=args.save_anomaly_maps,
                                        is_gpu = args.is_gpu)

# run experiment
experiment.run_experiment()
