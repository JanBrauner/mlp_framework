import data_providers as data_providers
import numpy as np
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
from model_architectures import ConvolutionalNetwork, create_model


# =============================================================================
# defaults: gamma_factor=1, rot_angle=0, shear_angle=0, translate_distance=0, 
#                  scale_factor=1,
# =============================================================================


default kernel size for CE is 4. Lets see how stuff like this can be implemented nicely (variable kernel size)
-- maybe dont use defaults at all, so that everything always throws an error if it wasnt provided explicitly?

defaults in CE paper: 
Learning_rate: 0.0002
beta1=0.5


add: 
    args.model_name
    
change name: 
    translate distance -> translate factor

augment = False


if args.augment:
    augmentations = [transforms.RandomAffine(degrees=args.rot_angle, translate=args.translate_factor, 
                                        scale=(1/args.scale_factor, args.scale_factor),
                                        shear=args.shear_angle)]
    # these augmentations are often used apparently:
#                transforms.RandomCrop(32, padding=4),
#                transforms.RandomHorizontalFlip(),
    
else:
    augmentations = None
    


args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

from torchvision import transforms
import torch

torch.manual_seed(seed=args.seed) # sets pytorch's seed

# create datasets
train_data, val_data, test_data, num_output_classes = data_providers.create_dataset(args):

# create model
model = model_architectures.create_model(args)


experiment = ExperimentBuilder(network_model=model,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data, device=device, args = args)  # build an experiment object
experiment_metrics, test_metrics = experiment.run_experiment()  # run experiment and return experiment metrics

