import numpy as np
import matplotlib.pyplot as plt
import torch

def show(img, cax=None):
    # show PIL image or torch Tensor
    if type(img) == torch.Tensor:
        npimg = img.numpy()
        npimg = np.transpose(npimg, (1,2,0))
    else:
        npimg = np.array(img)
    npimg = np.squeeze(npimg)
    if cax==None:
        _, cax = plt.subplots()
    cax.imshow(npimg, interpolation='nearest')
