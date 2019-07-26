# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.
"""
from __future__ import print_function
import pickle
import gzip
import numpy as np
import os
DEFAULT_SEED = 20112018
from PIL import Image
import os.path
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import math

import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import torchvision.transforms as transforms

class DataProvider(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(inputs.shape[0])
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def __next__(self):
        return self.next()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch

class MNISTDataProvider(DataProvider):
    """Data provider for MNIST handwritten digit images."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new MNIST data provider object.

        Args:
            which_set: One of 'train', 'val' or 'eval'. Determines which
                portion of the MNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'val', 'test'], (
            'Expected which_set to be either train, val or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 10
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # MLP_DATA_DIR environment variable should point to the data directory
        data_path = os.path.join(
            "data", 'mnist-{0}.npz'.format(which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        loaded = np.load(data_path)
        inputs, targets = loaded['inputs'], loaded['targets']
        inputs = inputs.astype(np.float32)
        # pass the loaded data to the parent class __init__
        super(MNISTDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(MNISTDataProvider, self).next()
        return inputs_batch, self.to_one_of_k(targets_batch)

    def to_one_of_k(self, int_targets):
        """Converts integer coded class target to 1 of K coded targets.

        Args:
            int_targets (ndarray): Array of integer coded class targets (i.e.
                where an integer from 0 to `num_classes` - 1 is used to
                indicate which is the correct class). This should be of shape
                (num_data,).

        Returns:
            Array of 1 of K coded targets i.e. an array of shape
            (num_data, num_classes) where for each row all elements are equal
            to zero except for the column corresponding to the correct class
            which is equal to one.
        """
        one_of_k_targets = np.zeros((int_targets.shape[0], self.num_classes))
        one_of_k_targets[range(int_targets.shape[0]), int_targets] = 1
        return one_of_k_targets

class EMNISTDataProvider(DataProvider):
    """Data provider for EMNIST handwritten digit images."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, flatten=False):
        """Create a new EMNIST data provider object.

        Args:
            which_set: One of 'train', 'val' or 'eval'. Determines which
                portion of the EMNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'val', 'test'], (
            'Expected which_set to be either train, val or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 47
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # MLP_DATA_DIR environment variable should point to the data directory
        data_path = os.path.join(
            "data", 'emnist-{0}.npz'.format(which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        loaded = np.load(data_path)
        print(loaded.keys())
        inputs, targets = loaded['inputs'], loaded['targets']
        inputs = inputs.astype(np.float32)
        if flatten:
            inputs = np.reshape(inputs, newshape=(-1, 28*28))
        else:
            inputs = np.reshape(inputs, newshape=(-1, 1, 28, 28))
        inputs = inputs / 255.0
        # pass the loaded data to the parent class __init__
        super(EMNISTDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

    def __len__(self):
        return self.num_batches

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(EMNISTDataProvider, self).next()
        return inputs_batch, self.to_one_of_k(targets_batch)

    def to_one_of_k(self, int_targets):
        """Converts integer coded class target to 1 of K coded targets.

        Args:
            int_targets (ndarray): Array of integer coded class targets (i.e.
                where an integer from 0 to `num_classes` - 1 is used to
                indicate which is the correct class). This should be of shape
                (num_data,).

        Returns:
            Array of 1 of K coded targets i.e. an array of shape
            (num_data, num_classes) where for each row all elements are equal
            to zero except for the column corresponding to the correct class
            which is equal to one.
        """
        one_of_k_targets = np.zeros((int_targets.shape[0], self.num_classes))
        one_of_k_targets[range(int_targets.shape[0]), int_targets] = 1
        return one_of_k_targets


class MetOfficeDataProvider(DataProvider):
    """South Scotland Met Office weather data provider."""

    def __init__(self, window_size, batch_size=10, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new Met Office data provider object.

        Args:
            window_size (int): Size of windows to split weather time series
               data into. The constructed input features will be the first
               `window_size - 1` entries in each window and the target outputs
               the last entry in each window.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        data_path = os.path.join(
            os.environ['DATASET_DIR'], 'HadSSP_daily_qc.txt')
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        raw = np.loadtxt(data_path, skiprows=3, usecols=range(2, 32))
        assert window_size > 1, 'window_size must be at least 2.'
        self.window_size = window_size
        # filter out all missing datapoints and flatten to a vector
        filtered = raw[raw >= 0].flatten()
        # normalise data to zero mean, unit standard deviation
        mean = np.mean(filtered)
        std = np.std(filtered)
        normalised = (filtered - mean) / std
        # create a view on to array corresponding to a rolling window
        shape = (normalised.shape[-1] - self.window_size + 1, self.window_size)
        strides = normalised.strides + (normalised.strides[-1],)
        windowed = np.lib.stride_tricks.as_strided(
            normalised, shape=shape, strides=strides)
        # inputs are first (window_size - 1) entries in windows
        inputs = windowed[:, :-1]
        # targets are last entry in windows
        targets = windowed[:, -1]
        super(MetOfficeDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

class CCPPDataProvider(DataProvider):

    def __init__(self, which_set='train', input_dims=None, batch_size=10,
                 max_num_batches=-1, shuffle_order=True, rng=None):
        """Create a new Combined Cycle Power Plant data provider object.

        Args:
            which_set: One of 'train' or 'val'. Determines which portion of
                data this object should provide.
            input_dims: Which of the four input dimension to use. If `None` all
                are used. If an iterable of integers are provided (consisting
                of a subset of {0, 1, 2, 3}) then only the corresponding
                input dimensions are included.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        data_path = os.path.join(
            os.environ['DATASET_DIR'], 'ccpp_data.npz')
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # check a valid which_set was provided
        assert which_set in ['train', 'val'], (
            'Expected which_set to be either train or val '
            'Got {0}'.format(which_set)
        )
        # check input_dims are valid
        if not input_dims is not None:
            input_dims = set(input_dims)
            assert input_dims.issubset({0, 1, 2, 3}), (
                'input_dims should be a subset of {0, 1, 2, 3}'
            )
        loaded = np.load(data_path)
        inputs = loaded[which_set + '_inputs']
        if input_dims is not None:
            inputs = inputs[:, input_dims]
        targets = loaded[which_set + '_targets']
        super(CCPPDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)


class AugmentedMNISTDataProvider(MNISTDataProvider):
    """Data provider for MNIST dataset which randomly transforms images."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, transformer=None):
        """Create a new augmented MNIST data provider object.

        Args:
            which_set: One of 'train', 'val' or 'test'. Determines which
                portion of the MNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
            transformer: Function which takes an `inputs` array of shape
                (batch_size, input_dim) corresponding to a batch of input
                images and a `rng` random number generator object (i.e. a
                call signature `transformer(inputs, rng)`) and applies a
                potentiall random set of transformations to some / all of the
                input images as each new batch is returned when iterating over
                the data provider.
        """
        super(AugmentedMNISTDataProvider, self).__init__(
            which_set, batch_size, max_num_batches, shuffle_order, rng)
        self.transformer = transformer

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(
            AugmentedMNISTDataProvider, self).next()
        transformed_inputs_batch = self.transformer(inputs_batch, self.rng)
        return transformed_inputs_batch, targets_batch




class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, which_set,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.which_set = which_set  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        rng = np.random.RandomState(seed=0)

        train_sample_idx = rng.choice(a=[i for i in range(50000)], size=47500, replace=False)
        val_sample_idx = [i for i in range(50000) if i not in train_sample_idx]

        if self.which_set is 'train':
            self.data = []
            self.labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.labels += entry['labels']
                else:
                    self.labels += entry['fine_labels']
                fo.close()

            self.data = np.concatenate(self.data)

            self.data = self.data.reshape((50000, 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
            self.data = self.data[train_sample_idx]
            self.labels = np.array(self.labels)[train_sample_idx]
            print(which_set, self.data.shape)
            print(which_set, self.labels.shape)

        elif self.which_set is 'val':
            self.data = []
            self.labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.labels += entry['labels']
                else:
                    self.labels += entry['fine_labels']
                fo.close()

            self.data = np.concatenate(self.data)
            self.data = self.data.reshape((50000, 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
            self.data = self.data[val_sample_idx]
            self.labels = np.array(self.labels)[val_sample_idx]
            print(which_set, self.data.shape)
            print(which_set, self.labels.shape)

        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.data = entry['data']
            if 'labels' in entry:
                self.labels = entry['labels']
            else:
                self.labels = entry['fine_labels']
            fo.close()
            self.data = self.data.reshape((10000, 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
            self.labels = np.array(self.labels)
            print(which_set, self.data.shape)
            print(which_set, self.labels.shape)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.which_set
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

class InpaintingDataset(data.Dataset):
    
    def __init__(self, which_set, transformer, rng, args, inv_normalisation_transform, precompute_patches=True): #, patch_rejection_threshold):

        # check a valid which_set was provided
        assert which_set in ['train', 'val', 'test'], (
            'Expected which_set to be either train, val or test '
            'Got {0}'.format(which_set)
        )

        self.which_set = which_set  # train, val or test set
        self.rng = rng       
        self.transformer = transformer
        self.inv_normalisation_transform = inv_normalisation_transform # transform that undoes the normalisation. Needed if target images should be created for classification (256 classes)
        
        self.patch_size = args.patch_size
        self.patch_location = args.patch_location_during_training
        self.mask_size = args.mask_size

        self.patch_mode = args.patch_mode
        self.target_format_for_classification = True if args.task == "classification" else False # returns the target (content of mask) as (CxHxW) tensor with integer values between 0 and 255
        
        self.scale_image = args.scale_image # tuple that contains the scaling factors for each of the two image dimensions. None if no scaling is used.

        try:
            self.data_format = args.data_format # Options: 
                            # "inpainting": input image/patch is masked out, and the target is the content of the masked image. 
                            # "autoencoding": input image/patch == output image, no masking.
        # Note: this should definitely be a separate class :-)
        except: # default value if the config doesn't have a value for this
            self.data_format = "inpainting"
        
        self.precompute_patches = precompute_patches # This is true by default. False is only required to make visualisation of val and test set faster
#        self.patch_rejection_threshold = patch_rejection_threshold

        # create list of all images in current dataset
        self.image_path = os.path.join(self.image_base_path, which_set)
        self.image_list = os.listdir(self.image_path) #directory may only contain the image files, no other files or directories
        assert len(self.image_list) > 0, "source directory doesn't contain image files"

        # debugging mode sets the dataset size to 50, so we can run the whole experiment locally.
        debug_mode = args.debug_mode
        if debug_mode:
            self.image_list = self.image_list[0:50]
            
        # calculate central regions slices
        # self.patch_slice = self.create_central_region_slice((1,1024,1024), self.patch_size) ### this can be done if images have all equal size
        self.mask_slice = self.create_central_region_slice((1,)+tuple(self.patch_size), self.mask_size)

        # Precompute the patch positions for val and test set. The patch positions in the val set and the test set should always be the same, so the vval/test set is not easier/harder on certain epochs
        if (self.which_set == "val" or self.which_set == "test") and self.patch_mode and self.precompute_patches:
            self.rng = np.random.RandomState(seed=0)
            self.compute_patch_positions()

    def compute_patch_positions(self):
        # Compute a random patch location for every image in the dataset
        self.top_left_corners = []
        for image_name in self.image_list:
            # load image
            full_image = Image.open(os.path.join(self.image_path, image_name))
            
            # potentially scale image
            if self.scale_image != None:
                scaled_image_size = tuple(int(x*self.scale_image[dim]) for dim, x in enumerate(full_image.size))
                full_image = full_image.resize(scaled_image_size)
                
                # if full image is too small now to house even one patch, make it bigger again: This shouldn't happen as long as the images have a certain size, but I saw a bug on the cluster that indicated that is does happen....
                min_ratio_full_image_to_patch = min(full_image.size[0]/self.patch_size[0], full_image.size[1]/self.patch_size[1])
                if min_ratio_full_image_to_patch <= 1:
                    scaled_image_size = tuple(int(x/min_ratio_full_image_to_patch + 1) for x in full_image.size)
                    full_image = full_image.resize(scaled_image_size)
            
            # transform image            
            full_image = self.transformer(full_image)
            
            top_left_corner = (self.rng.randint(0,full_image.shape[1]-self.patch_size[0]), 
                       self.rng.randint(0,full_image.shape[2]-self.patch_size[1]))
            self.top_left_corners.append(top_left_corner) # location of top-left corner of patch in dimensions 1 and 2 of the input tensor


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
        """
        
        # load image
        full_image = Image.open(os.path.join(self.image_path, self.image_list[index]))
        
        # potentially scale image
        if self.scale_image != None:
            scaled_image_size = tuple(int(x*self.scale_image[dim]) for dim, x in enumerate(full_image.size))
            full_image = full_image.resize(scaled_image_size)
        
            # if full image is too small now to house even one patch, make it bigger again: This shouldn't happen as long as the images have a certain size, but I saw a bug on the cluster that indicated that is does happen....
            min_ratio_full_image_to_patch = min(full_image.size[0]/self.patch_size[0], full_image.size[1]/self.patch_size[1])
            if min_ratio_full_image_to_patch <= 1:
                scaled_image_size = tuple(int(x/min_ratio_full_image_to_patch + 1) for x in full_image.size)
                full_image = full_image.resize(scaled_image_size)           


        # transform image
        full_image = self.transformer(full_image)
        
        if self.patch_mode: # if a image patch should be returned
            # create patch, but the patch will be called "image" for consistency with other Dataset classes
            if self.patch_location == "central":
                
                # image = full_image[self.patch_slice] ### this can be done if images have all equal size
                    
                # This is the version to calculate a new cetnral_region_slice for every image. This becomes necessary if input images vary in shape
                central_region_slice = self.create_central_region_slice(full_image.size(), self.patch_size)
                image = full_image[central_region_slice]      
                
            
            if self.patch_location == "random":
    #            comment these things back in if rejection of dark patches is still desired
    #            image_mean = -100000
    #            while image_mean <= patch_rejection_threshold:    
                if self.which_set == "train" or (not self.precompute_patches): # choose a new random patch position at every epoch

                    top_left_corner = (self.rng.randint(0,full_image.shape[1]-self.patch_size[0]), 
                                       self.rng.randint(0,full_image.shape[2]-self.patch_size[1])) # location of top-left corner of patch in dimensions 1 and 2 of the input tensor

                elif (self.which_set == "val" or self.which_set == "test") and self.precompute_patches: # take precomputed patch position that is constant over the epochs
                    top_left_corner = self.top_left_corners[index]
                    
                image = full_image[:,
                   top_left_corner[0]:top_left_corner[0]+self.patch_size[0],
                   top_left_corner[1]:top_left_corner[1]+self.patch_size[1]]
    #                image_mean = torch.mean(image)
        else:
            image = full_image
        
        if self.data_format == "inpainting":
            # create target image
            target_image = image[self.mask_slice].clone().detach()
                        
            # mask out central region in input image
            image[self.mask_slice] = 0 # note that zero is the dataset-wide mean value as images are centered
        elif self.data_format == "autoencoding":
            target_image = image.clone().detach()
        
        # format target image for classification task: (dimensions stay CxHxW,but every pixel needs to be an integer between 0 and 255)
        if self.target_format_for_classification:
            target_image = self.inv_normalisation_transform(target_image) # scales image to [0, 1]
            target_image = target_image*255 # scales image to [0, 255]
            target_image = target_image.type(torch.long)
           
        return image, target_image

    def create_central_region_slice(self, image_size, size_central_region):
        margins = ((image_size[1]-size_central_region[0])/2, 
                   (image_size[2]-size_central_region[1])/2) # size of margins in dimensions 1 and 2 (relative to the 3-D tensor) between the image borders and the patch borders
        
        central_region_slice = np.s_[:, 
                          math.ceil(margins[0]):math.ceil(image_size[1]-margins[0]), 
                          math.ceil(margins[1]):math.ceil(image_size[2]-margins[1])]
        return central_region_slice


    def __len__(self):
        return len(self.image_list)

class MiasHealthy(InpaintingDataset):
    """

    """    
    def __init__(self, which_set, transformer, rng, args, inv_normalisation_transform, precompute_patches=True): #, patch_rejection_threshold):
    
        # DATASET_DIR should point to root directory of data
        self.image_base_path = os.path.join(os.environ['DATASET_DIR'], "MiasHealthy") # path of the data set images
        print("Loading data from: ", self.image_base_path)
        
        super(MiasHealthy, self).__init__(
                which_set=which_set, 
                transformer=transformer, 
                rng=rng,
                args = args,
                inv_normalisation_transform=inv_normalisation_transform,
                precompute_patches=precompute_patches) #, patch_rejection_threshold):

class GoogleStreetView(InpaintingDataset):
    """

    """
    
    def __init__(self, which_set, transformer, rng, args, inv_normalisation_transform, precompute_patches=True): #, patch_rejection_threshold):


        # DATASET_DIR should point to root directory of data
        self.image_base_path = os.path.join(os.environ['DATASET_DIR'], "GoogleStreetView") # path of the data set images
        print("Loading data from: ", self.image_base_path)

        super(GoogleStreetView, self).__init__(
                which_set=which_set, 
                transformer=transformer, 
                rng=rng,
                args = args,
                inv_normalisation_transform=inv_normalisation_transform,
                precompute_patches=precompute_patches) #, patch_rejection_threshold):
                
class DescribableTextures(InpaintingDataset):
    """

    """
    
    def __init__(self, which_set, transformer, rng, args, inv_normalisation_transform, precompute_patches=True): #, patch_rejection_threshold):


        # DATASET_DIR should point to root directory of data
        self.image_base_path = os.path.join(os.environ['DATASET_DIR'], "DescribableTextures") # path of the data set images
        print("Loading data from: ", self.image_base_path)

        super(DescribableTextures, self).__init__(
                which_set=which_set, 
                transformer=transformer, 
                rng=rng,
                args = args,
                inv_normalisation_transform=inv_normalisation_transform,
                precompute_patches=precompute_patches) #, patch_rejection_threshold):

class DatasetWithAnomalies(InpaintingDataset): # the only thing it inherits is get_central_region_slice

    def __init__(self, which_set, transformer, args, inv_normalisation_transform=None):

        # check a valid which_set was provided
        assert which_set in ['train', 'val', 'test'], (
            'Expected which_set to be either train, val or test '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set  # train, val or test set
        
        print("Loading data from: ", self.image_base_path)
        self.image_path = os.path.join(self.image_base_path, which_set, "images")
        self.image_list = os.listdir(self.image_path) #directory may only contain the image files, no other files or directories
        assert len(self.image_list) > 0, "source directory doesn't contain image files"

        self.label_image_path = os.path.join(self.image_base_path, which_set, "label_images") # folder is supposed to contain the binary label images, same names as the input images


        self.transformer = transformer
        self.inv_normalisation_transform = inv_normalisation_transform # transform that undoes the normalisation. Needed if target images should be created for classification (256 classes)        

        self.label_image_transformer = transforms.ToTensor() # this is the only transformation label images need
        
        self.patch_size = args.patch_size
        self.mask_size = args.mask_size
        self.patch_stride = args.AD_patch_stride        
        self.scale_image = args.scale_image
        
        try:
            self.data_format = args.data_format # Options: 
                            # "inpainting": input image/patch is masked out, and the target is the content of the masked image. 
                            # "autoencoding": input image/patch == output image, no masking.
        # Note: this should definitely be a separate class :-)
        except: # default value if the config doesn't have a value for this
            self.data_format = "inpainting"
        
        self.target_format_for_classification = True if args.task == "classification" else False # returns the target (content of mask) as (CxHxW) tensor with integer values between 0 and 255

        # debugging mode sets the dataset size to around 10, so we can run the whole experiment locally.
        debug_mode = args.debug_mode
        if debug_mode:
            debug_image_idxs = list(range(0,len(self.image_list),int(len(self.image_list)/10))) # I want images from across the dataset (if it is order), but the same ones every time
            self.image_list = [self.image_list[debug_image_idx] for debug_image_idx in debug_image_idxs]
        
        self.sample_list, self.num_samples, self.image_sizes = self.create_sample_list()
            
        # calculate central regions slices
        self.mask_slice = self.create_central_region_slice((1,)+tuple(self.patch_size), self.mask_size)


    def create_sample_list(self):
        """
        Creates a list of all "samples" in the dataset: all valid sliding window positions in every image. Also determine the slice of the patch (relative to the ful image) that corresponds to every sample.
        Additionally, returns a list of the image sizes (of the full images) of every image.
        """
        sample_list = []
        full_image_sizes = []
        
        for image_idx, image_name in enumerate(self.image_list):
                
            # load image
            full_image = Image.open(os.path.join(self.image_path, image_name))

            # potentially scale image
            if self.scale_image != None:
                scaled_image_size = tuple(int(x*self.scale_image[dim]) for dim, x in enumerate(full_image.size))
                full_image = full_image.resize(scaled_image_size)
                
                # if full image is too small now to house even one patch, make it bigger again: This shouldn't happen as long as the images have a certain size, but I saw a bug on the cluster that indicated that is does happen....
                min_ratio_full_image_to_patch = min(full_image.size[0]/self.patch_size[0], full_image.size[1]/self.patch_size[1])
                if min_ratio_full_image_to_patch <= 1:
                    scaled_image_size = tuple(int(x/min_ratio_full_image_to_patch + 1) for x in full_image.size)
                    full_image = full_image.resize(scaled_image_size)           

            # transform image
            full_image = self.transformer(full_image)
            full_image_sizes.append(full_image.shape)
            assert (full_image.shape[1] > self.patch_size[0] and full_image.shape[2] > self.patch_size[1]), "Specified patch size larger than test image"
            
            num_windows_0 = 1 + ((full_image.shape[1] - self.patch_size[0]) // self.patch_stride[0]) # number of times that the sliding window fits into image dimension 0 (tensor dimension 1)
            num_windows_1 = 1 + ((full_image.shape[2] - self.patch_size[1]) // self.patch_stride[1]) # number of times that the sliding window fits into image dimension 1 (tensor dimension 2)
            
            # create one entry to the sample list for every sliding window position (axis 0 and 1 relatitve to 2-D image, not 3-D tensor)
            for pos_0 in range(num_windows_0):
                for pos_1 in range(num_windows_1):
                    current_slice = np.s_[:, 
                                          pos_0*self.patch_stride[0]:pos_0*self.patch_stride[0]+self.patch_size[0], 
                                          pos_1*self.patch_stride[1]:pos_1*self.patch_stride[1]+self.patch_size[1]]
                    sample_position = {
                            "image_idx": image_idx, 
                            "image_name": image_name, # maybe this isn't even needed
                            "slice": current_slice}
                    sample_list.append(sample_position)

        num_samples = len(sample_list)
            
        return sample_list, num_samples, full_image_sizes
     
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
        """
        
        # load the information about the next sample (not full image) from sample_list
        sample_info = self.sample_list[index]
        
        # load image
        full_image = Image.open(os.path.join(self.image_path, sample_info["image_name"]))

        # potentially scale image
        if self.scale_image != None:
            scaled_image_size = tuple(int(x*self.scale_image[dim]) for dim, x in enumerate(full_image.size))
            full_image = full_image.resize(scaled_image_size)

            # if full image is too small now to house even one patch, make it bigger again: This shouldn't happen as long as the images have a certain size, but I saw a bug on the cluster that indicated that is does happen....
            min_ratio_full_image_to_patch = min(full_image.size[0]/self.patch_size[0], full_image.size[1]/self.patch_size[1])
            if min_ratio_full_image_to_patch <= 1:
                scaled_image_size = tuple(int(x/min_ratio_full_image_to_patch + 1) for x in full_image.size)
                full_image = full_image.resize(scaled_image_size)  
                    
        # transform image
        full_image = self.transformer(full_image)
        
        # create patch (called "image" here, for consistency with other data providers)
        image = full_image[sample_info["slice"]]
        
        
        if self.data_format == "inpainting":
            # create target image
            target_image = image[self.mask_slice].clone().detach()
                        
            # mask out central region in input image
            image[self.mask_slice] = 0 # note that zero is the dataset-wide mean value as images are centered
        elif self.data_format == "autoencoding":
            target_image = image.clone().detach()
        
        
        # format target image for classification task: (dimensions stay CxHxW,but every pixel needs to be an integer between 0 and 255)
        if self.target_format_for_classification:
            target_image = self.inv_normalisation_transform(target_image) # scales image to [0, 1]
            target_image = target_image*255 # scales image to [0, 255]
            target_image = target_image.type(torch.long)
        
        # create mask slice relative to full image
        # Note: axis labels 0, 1, 2 are relatitve to 3-D tensor, not 2-D image
        # Note: PyTorch Dataloader doesn't deal with slices, but can work with dicts, so we will pass the relevant information in form of a dictionary
        slice_dict = {} 
        slice_1 = np.s_[sample_info["slice"][1].start + self.mask_slice[1].start : sample_info["slice"][1].start + self.mask_slice[1].stop]
        slice_dict["1_start"] = slice_1.start
        slice_dict["1_stop"] = slice_1.stop
        
        slice_2 = np.s_[sample_info["slice"][2].start + self.mask_slice[2].start : sample_info["slice"][2].start + self.mask_slice[2].stop]
        slice_dict["2_start"] = slice_2.start
        slice_dict["2_stop"] = slice_2.stop

        return image, target_image, sample_info["image_idx"], slice_dict
    
    def get_label_image(self, index):
        """
        Get binary label index that corresponds to the image image_list(index)
        
        Note: label image doesn't get scaled even if self.scale_image==True, that is on purpose (the evaluation is fairer if the anomaly maps get scaled to the size of the label image, not the other way around)
        """
        
        # load image
        full_label_image = Image.open(os.path.join(self.label_image_path, self.image_list[index]))
        
        # potentially scale image


        # transform image
        full_label_image = self.label_image_transformer(full_label_image)
    
        return full_label_image
    
    
    def __len__(self):
        return self.num_samples

class DescribableTexturesPathological(DatasetWithAnomalies):
    
    def __init__(self, which_set, transformer, args, inv_normalisation_transform=None):    
       
        version = args.anomaly_dataset_name # there are several versions of DTPathological with different lesions 
        self.image_base_path = os.path.join(os.environ['DATASET_DIR'], version) # path of the data set images
        
        super(DescribableTexturesPathological, self).__init__(
                which_set=which_set, 
                transformer=transformer,
                args = args, 
                inv_normalisation_transform=inv_normalisation_transform)

# =============================================================================
# 
# class MiasPathological(data.Dataset):
#     """
# 
#     """
# 
#     # Handling cluster data migration to scratch folder - use this when running on cluster
#     #image_path_base = os.path.join(os.environ['DATASET_DIR'], "RSNA_working_set") # path of the data set images
# 
# # =============================================================================
# #     # Use this for your local PC
#     image_base_path = os.path.join("data", "pathological_1") # path of the data set images
# 
# # =============================================================================
# 
# 
#     def __init__(self, which_set, # task,
#                  transformer,
#                  debug_mode=False, 
#                  patch_size=(256,256), mask_size=(64,64)):
# 
#         # check a valid which_set was provided
#         assert which_set in ['train', 'val', 'test'], (
#             'Expected which_set to be either train, val or test '
#             'Got {0}'.format(which_set)
#         )
# #        assert task in ["regression"], "Please enter valid task"
#         
#         self.which_set = which_set  # train, val or test set
# #        self.task = task
#         
#         self.patch_size = patch_size
#         self.mask_size = mask_size
#         self.transformer = transformer
# 
#         # create list of all images in current dataset
#         self.image_path = os.path.join(self.image_base_path, which_set, "images")
#         self.target_image_path = os.path.join(self.image_base_path, which_set, "target_images")
#         self.image_list = os.listdir(self.image_path).sort() #directory may only contain the image files, no other files or directories
#         self.target_image_list = os.listdir(self.target_image_path).sort() #directory may only contain the image files, no other files or directories
#         
#         assert len(self.image_list) > 0, "source directory doesn't contain image files"
#         assert len(self.image_list) == len(self.target_image_list)
#         for i,t in zip(self.image_list, self.target_image_list):
#             assert i[-6:-1] == t[-6:-1] # check that the last few symbols of the filenames are equal
# 
# 
#         # debugging mode sets the dataset size to 50, so we can run the whole experiment locally.
#         if debug_mode:
#             self.image_list = self.image_list[0:50]
#             self.target_image_list = self.target_image_list[0:50]
# 
# 
#     def get_input_patch(self, index, patch_location="central"):
#         """
#         Args:
#             index (int): Index
#             patch_location: can be the string "central" or a tupel of 2-D coordinates of the top-left corner
#         Returns:
#         """
#         
#         # load image
#         full_image = Image.open(os.path.join(self.image_path, self.image_list[index]))
# #        full_target_image = Image.open(os.path.join(self.target_image_path, self.target_image_list[index]))
# 
#         # transform image
#         full_image = self.transformer(full_image)
# #        full_target_image = self.transformer(full_target_image)
#         
#         # create patch, but the patch will be called "image" for consistency with other Dataset classes
#         if self.patch_location == "central":
#             central_region_slice = self.create_central_region_slice(full_image.size(), self.patch_size)
#             image = full_image[central_region_slice]      
#         
#         else:
#             top_left_corner = patch_location
#             image = full_image[:,
#                                top_left_corner[0]:top_left_corner[0]+self.patch_size[0],
#                                top_left_corner[1]:top_left_corner[1]+self.patch_size[1]]
#         
#         # create coordinates of central region of the image, to be masked out
#         central_region_slice = self.create_central_region_slice(image.size(), self.mask_size)
#         
# #        # create target image
# #        target_image = image[central_region_slice].clone().detach()
# #        
#         # mask out central region in input image
#         image[central_region_slice] = 0
#        
#         return image # , target_image
#     
#     def get_target(self, index):
#         '''
#         Get target segmentation of full image
#         '''
#         full_target_image = Image.open(os.path.join(self.target_image_path, self.target_image_list[index]))
#         full_target_image = self.transformer(full_target_image)
#         return full_target_image
# 
#     def create_central_region_slice(self, image_size, size_central_region):
#         margins = ((image_size[1]-size_central_region[0])/2, 
#                    (image_size[2]-size_central_region[1])/2) # size of margins in dimensions 1 and 2 (relative to the 3-D tensor) between the image borders and the patch borders
#         
#         central_region_slice = np.s_[:, 
#                           math.ceil(margins[0]):math.ceil(image_size[1]-margins[0]), 
#                           math.ceil(margins[1]):math.ceil(image_size[2]-margins[1])]
#         return central_region_slice
# 
# 
#     def __len__(self):
#         return len(self.image_list)
# 
# =============================================================================



        
def create_transforms(mn, sd, augmentations, args):
    # returns a list of standard transformations that get applied to the images of each set (train, val, test)  
    patch_mode = args.patch_mode
    image_height = args.image_height
    image_width = args.image_width
    
    if patch_mode: # patches get extracted within the Datset class, no need to resize images here
        standard_transforms = [transforms.ToTensor(),
                               transforms.Normalize(mn, sd)]
    else:
        standard_transforms = [transforms.Resize((image_height, image_width)), # resize the image to image_height x image_width
                               transforms.ToTensor(),
                               transforms.Normalize(mn, sd)]
    
    if augmentations is not None:
        transform_train = transforms.Compose(augmentations + standard_transforms)
    else:
        transform_train = transforms.Compose(standard_transforms)

    transform_test = transforms.Compose(standard_transforms)
    
    return transform_train, transform_test

def create_inv_normalise_transform(mn, sd):
    # for a given mean and sd, create transform that  undoes the normalisatin
    neg_of_mn_over_SD = [-x/y for x,y in zip(mn,sd)] # this has a bit weird syntax because it needs to deal with tupels and lists, since that's what transforms.normalise expects as input
    one_over_SD = [1./x for x in sd]
    inv_normalise = transforms.Normalize(neg_of_mn_over_SD, one_over_SD)
    return inv_normalise



def create_dataset(args, augmentations, rng, precompute_patches=True):
    """
    precompute_patches=True is only used during visualisation of val and test set (because it is faster not to precompute), but not during training.
    """
    if args.dataset_name == 'emnist':
        train_data = EMNISTDataProvider('train', batch_size=args.batch_size,
                                                       rng=rng, flatten=False)  # initialize our rngs using the argument set seed
        val_data = EMNISTDataProvider('val', batch_size=args.batch_size,
                                                     rng=rng, flatten=False)  # initialize our rngs using the argument set seed
        test_data = EMNISTDataProvider('test', batch_size=args.batch_size,
                                                      rng=rng, flatten=False)  # initialize our rngs using the argument set seed
        num_output_classes = train_data.num_classes
    
        num_output_classes = 666
    
        return train_data, val_data, test_data, num_output_classes

    elif args.dataset_name == 'cifar10':
        standard_transforms = [transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        
        if augmentations is not None:
            transform_train = transforms.Compose(augmentations + standard_transforms)
        else:
            transform_train = transforms.Compose(standard_transforms)
    
        transform_test = standard_transforms
    
        trainset = CIFAR10(root='data', which_set='train', download=True, transform=transform_train)
        train_data = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
        valset = CIFAR10(root='data', which_set='val', download=True, transform=transform_test)
        val_data = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
        testset = CIFAR10(root='data', which_set='test', download=True, transform=transform_test)
        test_data = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_output_classes = 10
        
        return train_data, val_data, test_data, num_output_classes
    
    elif args.dataset_name == 'cifar100':
        standard_transforms = [transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        
        if augmentations is not None:
            transform_train = transforms.Compose(augmentations + standard_transforms)
        else:
            transform_train = transforms.Compose(standard_transforms)
    
        transform_test = standard_transforms
        
        trainset = CIFAR100(root='data', which_set='train', download=True, transform=transform_train)
        train_data = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
        valset = CIFAR100(root='data', which_set='val', download=True, transform=transform_test)
        val_data = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
        testset = CIFAR100(root='data', which_set='test', download=True, transform=transform_test)
        test_data = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    
        num_output_classes = 100
        
        return train_data, val_data, test_data, num_output_classes
    
    
    
    
    
    elif args.dataset_name == 'MiasHealthy':
        # normalisation:
        if args.normalisation == "range-11": # if "range-11" is specified, always normalise to [-1,1]
            mn = (0.5,)
            sd = (0.5,)
        elif args.normalisation == "mn0sd1":
            if not args.patch_mode or args.patch_location_during_training == "random": # calculated offline: mean and SD for all training images (whole images)
                mn = (0.14581,)
                sd = (0.25929,)
            elif args.patch_mode and args.patch_location_during_training == "central": # calculated offline: mean and SD for all training images (central 256x256 patch)
                mn = (0.39865,)
                sd = (0.30890,)
        
        # transformations and augmentations        
        transform_train, transform_test = create_transforms(mn, sd, augmentations, args)

        inv_normalisation_transform = create_inv_normalise_transform(mn, sd) # required for creating target images for classification, see InpaintingDataset
#        # patches with mean value below this get rejected (so we don't sample too many black images)
#        patch_rejection_threshold = (args.patch_rejection_treshold/255 - mn)/sd

        kwargs_dataset= {"rng":rng,
                         "args":args,
                         "inv_normalisation_transform": inv_normalisation_transform,
                         "precompute_patches" : precompute_patches}
        kwargs_dataloader = {"batch_size": args.batch_size,
                             "num_workers": args.num_workers}
        
        trainset = MiasHealthy(which_set='train', transformer=transform_train, **kwargs_dataset)#, patch_rejection_threshold=patch_rejection_threshold)
        train_data = torch.utils.data.DataLoader(trainset, shuffle=True, **kwargs_dataloader)
    
        valset = MiasHealthy(which_set='val', transformer=transform_test, **kwargs_dataset) #, patch_rejection_threshold=patch_rejection_threshold)
        val_data = torch.utils.data.DataLoader(valset, shuffle=False, **kwargs_dataloader)
        testset = MiasHealthy(which_set='test', transformer=transform_test, **kwargs_dataset)#, patch_rejection_threshold=patch_rejection_threshold)
        test_data = torch.utils.data.DataLoader(testset, shuffle=False,  **kwargs_dataloader)
    
        num_output_classes = 666
        
        return train_data, val_data, test_data, num_output_classes




    elif args.dataset_name == 'GoogleStreetView':
        # calculate mean and SD for normalisation
        if args.normalisation == "mn0sd1":
            raise NotImplementedError
        elif args.normalisation == "range-11":
            mn = [0.5, 0.5, 0.5]
            sd = [0.5, 0.5, 0.5]

        # transformations and augmentations        
        transform_train, transform_test = create_transforms(mn, sd, augmentations, args)
                
        inv_normalisation_transform = create_inv_normalise_transform(mn, sd) # required for creating target images for classification, see InpaintingDataset

        kwargs_dataset= {"rng":rng,
                         "args":args,
                         "inv_normalisation_transform": inv_normalisation_transform,
                         "precompute_patches" : precompute_patches}
        kwargs_dataloader = {"batch_size": args.batch_size,
                             "num_workers": args.num_workers}
        
        trainset = GoogleStreetView(which_set='train', transformer=transform_train, **kwargs_dataset)#, patch_rejection_threshold=patch_rejection_threshold)
        train_data = torch.utils.data.DataLoader(trainset, shuffle=True, **kwargs_dataloader)
    
        valset = GoogleStreetView(which_set='val', transformer=transform_test, **kwargs_dataset) #, patch_rejection_threshold=patch_rejection_threshold)
        val_data = torch.utils.data.DataLoader(valset, shuffle=False, **kwargs_dataloader)
        
        testset = GoogleStreetView(which_set='test', transformer=transform_test, **kwargs_dataset)#, patch_rejection_threshold=patch_rejection_threshold)
        test_data = torch.utils.data.DataLoader(testset, shuffle=False, **kwargs_dataloader)
    
        num_output_classes = 666
        
        return train_data, val_data, test_data, num_output_classes
    
    
    
    
    
    elif args.dataset_name == "DescribableTextures":
        # calculate mean and SD for normalisation
        if args.normalisation == "mn0sd1":
            raise NotImplementedError
        elif args.normalisation == "range-11":
            mn = [0.5, 0.5, 0.5]
            sd = [0.5, 0.5, 0.5]

        # transformations and augmentations        
        transform_train, transform_test = create_transforms(mn, sd, augmentations, args)
        
        inv_normalisation_transform = create_inv_normalise_transform(mn, sd) # required for creating target images for classification, see InpaintingDataset

        
        kwargs_dataset= {"rng":rng,
                         "args":args,
                         "inv_normalisation_transform": inv_normalisation_transform,
                         "precompute_patches" : precompute_patches}
        kwargs_dataloader = {"batch_size": args.batch_size,
                             "num_workers": args.num_workers}
        
        trainset = DescribableTextures(which_set='train', transformer=transform_train, **kwargs_dataset)#, patch_rejection_threshold=patch_rejection_threshold)
        train_data = torch.utils.data.DataLoader(trainset, shuffle=True, **kwargs_dataloader)
    
        valset = DescribableTextures(which_set='val', transformer=transform_test, **kwargs_dataset) #, patch_rejection_threshold=patch_rejection_threshold)
        val_data = torch.utils.data.DataLoader(valset, shuffle=False, **kwargs_dataloader)
        
        testset = DescribableTextures(which_set='test', transformer=transform_test, **kwargs_dataset)#, patch_rejection_threshold=patch_rejection_threshold)
        test_data = torch.utils.data.DataLoader(testset, shuffle=False, **kwargs_dataloader)
    
        num_output_classes = 666
        
        return train_data, val_data, test_data, num_output_classes
    
def create_dataset_with_anomalies(args):
    anomaly_dataset_name = args.anomaly_dataset_name
    normalisation = args.normalisation
    batch_size = args.AD_batch_size
    num_workers = args.num_workers
    
    if "DTPathological" in anomaly_dataset_name: # there are several versions of DTPathological with different lesions 
        # calculate mean and SD for normalisation
        if normalisation == "mn0sd1":
            raise NotImplementedError
        elif normalisation == "range-11":
            mn = [0.5, 0.5, 0.5]
            sd = [0.5, 0.5, 0.5]
            
        transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mn, sd)])
        inv_normalisation_transform = create_inv_normalise_transform(mn, sd) # required for creating target images for classification, see InpaintingDataset

        kwargs_dataset = {"transformer":transformer, 
                          "args": args,
                          "inv_normalisation_transform": inv_normalisation_transform}
        val_dataset = DescribableTexturesPathological(which_set="val", **kwargs_dataset)
        test_dataset = DescribableTexturesPathological(which_set="test", **kwargs_dataset)

        
    ### data_loader
    val_data_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    
    return val_dataset, val_data_loader, test_dataset, test_data_loader 



