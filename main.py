import numpy as np
import torch
from torchvision import transforms

import data_providers as data_providers
import model_architectures
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder




# =============================================================================
# 
# # =============================================================================
# ### experiment settings
# args.experiment_name
# args.continue_from_epoch
# 
# # =============================================================================
# ### misc parameters
# args.seed
# args.task
# 
# # =============================================================================
# ### GPU settings
# args.use_gpu
# args.gpu_id
# 
# # =============================================================================
# ### model parameters
# args.model_name
# 
# # model parameters: convolutions
# args.kernel_size
# 
# # model parameters: encoder
# args.num_layers_enc
# args.num_channels_enc
# args.num_channels_progression_enc
# args.num_channels_bottleneck
# 
# # model parameters: decoder
# args.num_layers_dec
# args.num_channels_dec
# args.num_channels_progression_dec
# 
# # =============================================================================
# ### data parameters
# # data parameters: dataset
# args.dataset_name
# args.num_image_channels
# args.image_height
# args.image_width
# 
# # data parameters: misc
# args.debug_mode
# args.num_workers
# 
# # data parameters: augmentations
# args.augment = False
# args.gamma_factor
# args.rot_angle
# args.translate_factor
# args.scale_factor
# args.shear_angle
# 
# # data parameters: image patches
# args.patch_size
# args.patch_location_during_training
# 
# # data parameters: masking
# args.mask_size
# 
# # =============================================================================
# ### training parameters
# # training parameters: general
# args.batch_size
# args.loss
# args.num_epochs
# 
# # training parameters: optimiser
# args.learning_rate
# args.betas
# args.weight_decay_coefficient
# 
# # =============================================================================
# 
# 
# =============================================================================





# =============================================================================
# 
# ###### defaults
# default kernel size for CE is 4. Lets see how stuff like this can be implemented nicely (variable kernel size)
# -- maybe dont use defaults at all, so that everything always throws an error if it wasnt provided explicitly?
# 
# defaults in CE implementation: 
# args.learning_rate: 0.0002
# beta1=0.5

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

