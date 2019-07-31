from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import math

def get_aucroc(y_true, output):
    if torch.min(y_true.data) == 1 or torch.max(y_true.data) == 0:
        aucroc = np.nan # return nan if there are only examples of one type in the batch, because AUCROC is not defined then. 
    else:
        y_true = y_true.cpu().detach().numpy().flatten().astype(bool)
        output = output.cpu().detach().numpy().flatten()
        aucroc = roc_auc_score(y_true,output)
    return aucroc


def create_central_region_slice(image_size, size_central_region):
    """
    create slice (HxW) of the central region of an image (dimensions (HxW)), when the size of the central region is central_region_size (HxW)
    
    """
    margins = ((image_size[0]-size_central_region[0])/2, 
               (image_size[1]-size_central_region[1])/2) # size of margins in dimensions 1 and 2 (relative to the 3-D tensor) between the image borders and the patch borders
    
    central_region_slice = np.s_[math.ceil(margins[0]):math.ceil(image_size[0]-margins[0]), 
                                 math.ceil(margins[1]):math.ceil(image_size[1]-margins[1])]
    return central_region_slice
