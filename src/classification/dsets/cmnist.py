import os
import torch
from torchvision import datasets
from torch.utils.data import Dataset, TensorDataset, random_split
import numpy as np
from .vision import VisionDataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


class ourMNIST(VisionDataset):
    """
    original MNIST with dequantization.
    for our purposes, we have "ourMNIST" (black background) as y=0, and FlippedMNIST (white background) as y=1. so all labels for this class will be returned as y=0.
    """
    def __init__(self,
                config,
                split='train',
                transform=None, target_transform=None, load_in_mem=False,
                download=True, **kwargs):
        super(ourMNIST, self).__init__(config.training.data_dir)

        self.split = split
        self.perc = config.data.perc
        self.lam = 1e-6
        self.root = os.path.join(config.training.data_dir, 'mnist/')
        mnist = datasets.MNIST(self.root, train=True if self.split in ['train', 'val'] else False, download=True)  # don't apply transformations

        if split in ['train', 'val']:
            num_train = int(0.8 * len(mnist.train_data))
            train_idxs = np.random.choice(np.arange(len(mnist.train_data)), size=num_train, replace=False)
            val_idxs = np.setdiff1d(np.arange(len(mnist.train_data)), train_idxs)

            data_idxs = train_idxs if split == 'train' else val_idxs
            self.data = mnist.train_data[data_idxs]
            self.labels = mnist.train_labels[data_idxs]
        else:
            self.data = mnist.test_data
            self.labels = mnist.test_labels

    def __getitem__(self, index):

        # get anchor data points
        item = self.data[index]
        # label = self.labels[index]
        label = torch.zeros(1)
        item = item.view((-1, 784))

        return item, label

    def __len__(self):
        return len(self.data)


class FlippedMNIST(VisionDataset):
    """
    MNIST with background color flipped.
    for our purposes, we have "ourMNIST" (black background) as y=0, and FlippedMNIST (white background) as y=1. so all labels for this class will be returned as y=1.
    """
    def __init__(self,
                config,
                split='train',
                transform=None, target_transform=None, load_in_mem=False,
                download=True, **kwargs):
        super(FlippedMNIST, self).__init__(config.training.data_dir)

        self.split = split
        self.perc = config.data.perc
        self.lam = 1e-6
        self.root = os.path.join(config.training.data_dir, 'mnist/')
        mnist = datasets.MNIST(self.root, train=True if self.split in ['train', 'val'] else False, download=True)  # don't apply transformations

        if split in ['train', 'val']:
            num_train = int(0.8 * len(mnist.train_data))
            train_idxs = np.random.choice(np.arange(len(mnist.train_data)), size=num_train, replace=False)
            val_idxs = np.setdiff1d(np.arange(len(mnist.train_data)), train_idxs)

            data_idxs = train_idxs if split == 'train' else val_idxs
            data = mnist.train_data[data_idxs]
            labels = mnist.train_labels[data_idxs]
        else:
            data = mnist.test_data
            labels = mnist.test_labels

        self.data, self.labels = self.initialize_data_splits(data, labels, config.data.include_all)

    def initialize_data_splits(self, data, labels, include_all):
        """
        set aside a balanced number of classes for specified perc
        """
        if not include_all:
            n_examples = int(len(data) * self.perc)
            unique = torch.unique(labels)
            n_classes = len(unique)

            new_dset = []
            new_labels = []
            for class_label in unique:
                num_samples = n_examples // n_classes
                sub_y = labels[labels==class_label][0:num_samples] 
                sub_x = data[labels==class_label][0:num_samples]

                # add examples
                new_labels.append(sub_y)
                new_dset.append(sub_x)
            new_labels = torch.cat(new_labels)
            new_dset = torch.cat(new_dset)
        else:
            new_dset = data
            new_labels = labels
        # apply reverse black/white background
        new_dset = (255 - new_dset)

        return new_dset, new_labels

    def __getitem__(self, index):

        # get anchor data points
        item = self.data[index]
        # label = self.labels[index]
        label = torch.ones(1)
        item = item.view((-1, 784))

        return item, label

    def __len__(self):
        return len(self.data)


class MNISTSubset(ourMNIST):
    '''
    MNIST with only subset of the digits (0,7)
    for our purposes, all digits with a class label of 0 will be labeled as y=0. all digits with class label of 7 will be labeled as y=1.
    '''
    def __init__(self,
                config,
                split='train',
                transform=None, target_transform=None, load_in_mem=False,
                download=True, **kwargs):
        super(MNISTSubset, self).__init__(
                config, 
                split=split,
                transform=transform, 
                target_transform=target_transform, 
                load_in_mem=load_in_mem,
                download=download)


        mnist = datasets.MNIST(self.root, train=True if (self.split != 'test')  else False, download=True)  # don't apply transformations yet
        # list of digits to include
        self.digits = torch.Tensor(config.data.digits)
        # digit_percs[i] = what % of the dataset digits[i] should make up
        self.digit_percs = torch.Tensor(config.data.digit_percs)
        max_perc_idx = torch.argmax(self.digit_percs)

        # get correct data split
        if split != 'test':
            data = mnist.train_data
            targets = mnist.train_labels
        else:
            data = mnist.test_data
            targets = mnist.test_labels
        n_samples_needed = sum(targets == self.digits[max_perc_idx]) // self.digit_percs[max_perc_idx]
        subset_idxs = []
        for digit, perc in zip(self.digits, self.digit_percs):
            digit_idxs = torch.where(targets == digit)[0]
            
            # balanced digit split for test/val set; split by digit_percs for train
            if not config.data.includ
            e_all:
                n_digit_samples = int(perc * n_samples_needed) if split == 'train' else int(n_samples_needed.item() // len(self.digits))
            else:
                n_digit_samples = len(digit_idxs)
            digit_idxs = digit_idxs[:n_digit_samples]
            subset_idxs.extend(digit_idxs)
        
        self.data = data[subset_idxs]
        self.labels = targets[subset_idxs]

        if split == 'train' or split == 'val':
            num_train = int(0.8 * len(self.data))

            train_idxs = np.random.choice(np.arange(len(self.data)), size=num_train, replace=False)
            val_idxs = np.setdiff1d(np.arange(len(self.data)), train_idxs)

            data_idxs = train_idxs if split == 'train' else val_idxs
            self.data = self.data[data_idxs]
            self.labels = self.labels[data_idxs]

    def __getitem__(self, index):

        # get anchor data points
        item = self.data[index]
        label = self.labels[index]
        label = (label==7).float()  # TODO: hardcoded

        # dequantize input
        # (TODO: maybe this won't work out of the box without rng)
        item = item.view((-1, 784))

        return item, label


class FlippedMNISTSubset(ourMNIST):
    '''
    Flipped MNIST with only subset of the digits
    '''
    def __init__(self,
                config,
                split='train',
                transform=None, target_transform=None, load_in_mem=False,
                download=True, **kwargs):
        super(FlippedMNISTSubset, self).__init__(
                config, 
                split=split,
                transform=transform, 
                target_transform=target_transform, 
                load_in_mem=load_in_mem,
                download=download)


        # list of digits to include
        self.digits = torch.Tensor(config.data.digits)
        # digit_percs[i] = what % of the dataset digits[i] should make up
        self.digit_percs = torch.Tensor(config.data.digit_percs)

        mnist = datasets.MNIST(self.root, train=True if self.split != 'test' else False, download=True)
        # get correct data split
        if split != 'test':
            data = mnist.train_data
            targets = mnist.train_labels
        else:
            data = mnist.test_data
            targets = mnist.test_labels
        self.data, self.labels = self.initialize_data_splits(
            data, targets, split, config.data.include_all)

    def initialize_data_splits(self, data, targets, split, include_all):
        
        # select datapoints with desired digits
        digit_idxs = [] 
        for digit in self.digits:
            digit_idxs.extend(torch.where(targets == digit)[0])
        data = data[digit_idxs]
        labels = targets[digit_idxs]
        
        # divide into train and val sets
        if split == 'train' or split == 'val':
            num_train = int(0.8 * len(data))
            train_idxs = np.random.choice(np.arange(len(data)), size=num_train, replace=False)
            val_idxs = np.setdiff1d(np.arange(len(data)), train_idxs)
            data_idxs = train_idxs if split == 'train' else val_idxs
            data = data[data_idxs]
            labels = labels[data_idxs]
        
        if not include_all:
            # cut down dataset size and construct splits
            max_perc_idx = torch.argmax(self.digit_percs)
            total_samples_available = len(labels)
            n_samples_needed = min(int(float(self.perc) * total_samples_available), sum(labels == self.digits[max_perc_idx]).item() // self.digit_percs[max_perc_idx])

            subset_idxs = []
            for digit, digit_perc in zip(self.digits, self.digit_percs):
                digit_idxs = torch.where(labels == digit)[0]
                # balanced digit split for test/val set; split by digit_percs for train
                n_digit_samples = int(digit_perc * n_samples_needed) if split == 'train' else int(n_samples_needed // len(self.digits))
                digit_idxs = digit_idxs[:n_digit_samples]
                subset_idxs.extend(digit_idxs)
            data = data[subset_idxs]
            labels = labels[subset_idxs]
        # apply reverse black/white background
        data = (255 - data)

        return data, labels

    def __getitem__(self, index):

        # get anchor data points
        item = self.data[index]
        label = self.labels[index]
        label = (label==7).float()  # TODO: hardcoded

        # dequantize input
        # (TODO: maybe this won't work out of the box without rng)
        item = item.view((-1, 784))

        return item, label