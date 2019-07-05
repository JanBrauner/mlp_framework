import numpy as np
import torch
from torchvision import transforms

import data_providers as data_providers
import model_architectures
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder




# =============================================================================
# defaults: gamma_factor=1, rot_angle=0, shear_angle=0, translate_distance=0, 
#                  scale_factor=1,
# =============================================================================


# hparams for model
args.model_name

args.batch_size, 
args.num_image_channels, 
args.image_height, 
args.image_width

args.num_layers_enc
args.num_channels_enc
args.num_channels_progression_enc
args.kernel_size
args.num_channels_bottleneck

args.num_layers_dec
args.num_channels_dec
args.num_channels_progression_dec


# hparams for data_providers
args.dataset_name
args.num_workers
args.patch_size,
args.patch_location
args.mask_size


# hparams for experiment builder
args.experiment_name
args.learning_rate
args.betas
args.weight_decay_coefficient
args.task
args.loss
args.num_epochs


# hparams for main
args.augment = False
args.seed
args.rot_angle
args.translate_factor
args.scale_factor
args.shear_angle

# hparams from old
args.debug_mode = True 
args.use_gpu
args.continue_from_epoch

# =============================================================================
# 
# ###### defaults
# default kernel size for CE is 4. Lets see how stuff like this can be implemented nicely (variable kernel size)
# -- maybe dont use defaults at all, so that everything always throws an error if it wasnt provided explicitly?
# 
# defaults in CE implementation: 
# args.learning_rate: 0.0002
# beta1=0.5
# num_channels_progression = [1,1,2,4,8]
# 
#     
# change name: 
#     translate distance -> translate factor
# 
# 
# =============================================================================


args, device = get_args()  # get arguments from command line

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
train_data, val_data, test_data, num_output_classes = data_providers.create_dataset(args)

# create model
model = model_architectures.create_model(args)

# build experiment
experiment = ExperimentBuilder(network_model=model,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data, device=device, args = args)

# run experiment and return experiment metrics
experiment_metrics, test_metrics = experiment.run_experiment()  

