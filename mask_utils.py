import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def random_blob(img_size, num_iter, threshold, maximum_blob_size, fixed_init=None):
    """Generates masks with random connected blobs.
    Parameters
    ----------
    img_size : see single_random_mask
    num_iter : int
        Number of iterations to expand random blob for. This number controls the size of the blob, I feel.
    threshold : float
        Number between 0 and 1. Probability of keeping a pixel hidden. This number somehow controls the shape of the blob
    fixed_init : tuple of ints or None
        If fixed_init is None, central position of blob will be sampled
        randomly, otherwise expansion will start from fixed_init. E.g.
        fixed_init = (6, 12) will start the expansion from pixel in row 6,
        column 12.
    """
    _, img_height, img_width = img_size
    # Defines the shifts around the central pixel which may be unmasked
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    if fixed_init is None:
        # Sample random initial position, but not on the fringes of the image
        init_pos = np.random.randint(128, img_height - 128), np.random.randint(128, img_width - 128)
    else:
        init_pos = (fixed_init[0], fixed_init[1])
    # Initialize mask and make init_pos visible
    mask = torch.zeros(1, 1, *img_size[1:])
    mask[0, 0, init_pos[0], init_pos[1]] = 1.
    # Initialize the list of seed positions
    seed_positions = [init_pos]
    # Randomly expand blob

    for i in range(num_iter):
        next_seed_positions = []
        for seed_pos in seed_positions:
            # Sample probability that neighboring pixel will be visible
            prob_visible = np.random.rand(len(neighbors))
            for j, neighbor in enumerate(neighbors):
                if prob_visible[j] > threshold:
                    current_h, current_w = seed_pos
                    shift_h, shift_w = neighbor
                    # Ensure new height stays within image boundaries
                    new_h = max(min(current_h + shift_h, img_height - 1), 0)
                    # Ensure new width stays within image boundaries
                    new_w = max(min(current_w + shift_w, img_width - 1), 0)
                    if mask[0, 0, new_h, new_w] != 1.: # added this is the avoid re-selecting pixels that are already positive as seeds, because that takes too long
                        # Update mask
                        mask[0, 0, new_h, new_w] = 1.
                        # Add new position to list of seeds
                        next_seed_positions.append((new_h, new_w))
        seed_positions = next_seed_positions
    
    # Calculate stats about mask size: How far does the mask extent across dimensions 0 and 1 of the image?
    mask_sum_0 = torch.sum(mask[0,0,:,:], dim = 0)
    mask_sum_1 = torch.sum(mask[0,0,:,:], dim = 1)
    mask_size_1 = torch.max(mask_sum_0) - torch.min(mask_sum_0)
    mask_size_0 = torch.max(mask_sum_1) - torch.min(mask_sum_1)
    mask_size = (mask_size_0, mask_size_1)
    return mask, mask_size


def multi_random_blobs(img_size, max_num_blobs, iter_range, threshold, maximum_blob_size):
    """Generates masks with multiple random connected blobs.
    Parameters
    ----------
    max_num_blobs : int
        Maximum number of blobs. Number of blobs will be sampled between 1 and
        max_num_blobs
    iter_range : (int, int)
        Lower and upper bound on number of iterations to be used for each blob.
        This will be sampled for each blob.
    threshold : float
        Number between 0 and 1. Probability of keeping a pixel hidden.
    """
    mask = torch.zeros(1, 1, *img_size[1:])
    mask_sizes = []
    # Sample number of blobs
    num_blobs = np.random.randint(1, max_num_blobs + 1)
    for _ in range(num_blobs):
        num_iter = np.random.randint(iter_range[0], iter_range[1])
        temp_mask, mask_size = random_blob(img_size, num_iter, threshold, maximum_blob_size)
        mask += temp_mask
        mask_sizes.append(mask_size)
    mask[mask > 0] = 1.
    return mask, mask_sizes


def batch_multi_random_blobs(img_size, max_num_blobs, iter_range, threshold,
                             batch_size, maximum_blob_size):
    """Generates batch of masks with multiple random connected blobs."""
    mask = torch.zeros(batch_size, 1, *img_size[1:])
    mask_sizes = []
    for i in range(batch_size):
        mask[i], current_mask_sizes = multi_random_blobs(img_size, max_num_blobs, iter_range, threshold, maximum_blob_size)
        mask_sizes = mask_sizes + current_mask_sizes
    return mask, mask_sizes
