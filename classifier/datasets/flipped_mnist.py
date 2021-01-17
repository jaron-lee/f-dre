import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from .looping import LoopingDataset


class SplitEncodedMNIST(Dataset):
    """ 
    dataset that returns (ref_z, biased_z) when iterated through via dataloader
    (need to specify targets upon dataloading)
    """
    def __init__(self, args, split='train'):

        self.args = args
        self.perc = args.perc
        self.ref_dset = self.load_dataset(split, 'cmnist')
        self.biased_dset = self.load_dataset(split, 'mnist')

    def load_dataset(self, split, variant='mnist'):
        record = np.load(os.path.join(self.args.data_dir, 'maf_{}_{}_z.npz'.format(split, variant)))
        zs = torch.from_numpy(record['z']).float()
        ys = record['y']
        d_ys = torch.from_numpy(record['d_y']).float()

        # Truncate biased test/val set to be same size as reference val/test sets
        if (split == 'test' or split == 'val') and variant == 'mnist':
            # len(self.ref_dset) is always <= len(self.biased_dset)
            zs = zs[:len(self.ref_dset)]
            d_ys = d_ys[:len(self.ref_dset)]
        dataset = TensorDataset(zs, d_ys)
        dataset = LoopingDataset(dataset)
        return dataset
    
    def __getitem__(self, index):
        ref_z, _ = self.ref_dset[index]
        biased_z, _ = self.biased_dset[index]

        #TODO: eventually also return attr label in addition to ref/bias label?
        return (ref_z, biased_z)
    
    def __len__(self):
        return len(self.ref_dset) + len(self.biased_dset)