import numpy as np
import torch
import torchvision
from torchvision import transforms

import data_providers as data_providers
import model_architectures
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
from torch.utils.tensorboard import SummaryWriter

#args, device = get_args()  # get arguments from command line
args, device = get_args("CE_test") # for loca debugging

if args.augment:
    augmentations = [transforms.RandomAffine(degrees=args.rot_angle, translate=args.translate_factor, 
                                        scale=(1/args.scale_factor, args.scale_factor),
                                        shear=args.shear_angle)]
    # these augmentations are often used apparently:
#                transforms.RandomCrop(32, padding=4),
#                transforms.RandomHorizontalFlip(),
    
else:
    augmentations = None

# set random seeds
rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(seed=args.seed)

# create datasets
train_data, val_data, test_data, num_output_classes = data_providers.create_dataset(args, augmentations, rng)
images, labels = next(iter(train_data))
grid = torchvision.utils.make_grid(images)


# create model
model = model_architectures.create_model(args)
writer = SummaryWriter()
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()

# =============================================================================
# 
# # build experiment
# experiment = ExperimentBuilder(network_model=model,
#                                     train_data=train_data, val_data=val_data,
#                                     test_data=test_data, device=device, args=args)
# 
# # run experiment and return experiment metrics
# experiment_metrics, test_metrics = experiment.run_experiment()  
# =============================================================================