import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

def show(img, cax=None):
    # show PIL image or torch Tensor
    if type(img) == torch.Tensor:
        npimg = img.detach().numpy()
        npimg = np.transpose(npimg, (1,2,0))
    else:
        npimg = np.array(img)
    npimg = np.squeeze(npimg)
    if cax==None:
        _, cax = plt.subplots()
    cax.imshow(npimg, interpolation='nearest')
    
    return cax


def image_grid_with_groups(*image_groups, grid_parameters):
    """ 
    Create image grid from several groups of images. Each image group is one line.
    For example, this can be used to display three lines of images:
        masked inputs
        outputs
        original images
        
    image_groups: 4-D Tensor (0-th dimension is used to split into images), or list of PIL images/Tensors
    grid_parameters: parameters as passed into make_grid
    """
    fig, ax = plt.subplots(nrows=len(image_groups))
    for idx, images in enumerate(image_groups):
        grid = torchvision.utils.make_grid(images, **grid_parameters)
#        grids.append(grid)
        if len(image_groups) > 1:
            cax = ax[idx]
        else:
            cax = ax
        show(grid, cax)
