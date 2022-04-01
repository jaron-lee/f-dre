#!/usr/bin/env python
# coding: utf-8



import os
import sys
import time
import shutil
import logging
import argparse
import yaml
import traceback
import math
import copy

import pandas as pd
import numpy as np
from scipy.special import expit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import pairwise_distances
sns.set_style('white')
sns.set_context('poster')

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
from flows.models.maf import MAF



## Helper functions

def logsumexp_1p(s):
    # numerically stable implementation of log sigmoid via logsumexp
    # NOTE: this assumes that you feed in -s for the positive probabilities
    if len(s.size()) == 2:
        s = s.squeeze()
    x = torch.stack([s, torch.zeros_like(s)], -1)
    val, _ = torch.max(x, 1)
    val = val.repeat(2,1).T
    logsigmoid = torch.clamp(s, 0) + torch.log(
        torch.sum(torch.exp(x - val), 1))

    return -logsigmoid

def dict2namespace(config):
    namespace = argparse.Namespace()
    if isinstance(config, list):
        # from creating config files
        for i in range(len(config)):
            for key, value in config[i].items():
                if isinstance(value, dict):
                    new_value = dict2namespace(value)
                else:
                    new_value = value
                setattr(namespace, key, new_value)
    else:
        # vanilla training
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
    return namespace


## Normalizing Flow Model

def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
            betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
            eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))
        
@torch.no_grad()
def test(model, dataloader, epoch, args, config, device):
    model.eval()
    logprobs = []

    # unconditional model
    for data in dataloader:

        if len(data) == 1:
            x, y = data[0], None
        else:
            x, y = data
            y = y.to(device)
        x = x.to(device).view(x.shape[0], -1)
        log_px = model.module.log_prob(x)
        logprobs.append(log_px)
    logprobs = torch.cat(logprobs, dim=0).to(device)

    logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.var(0).sqrt() / math.sqrt(len(dataloader.dataset))
    output = 'Evaluate ' + (epoch != None)*'(epoch {}) -- '.format(epoch) + 'logp(x) = {:.3f} +/- {:.3f}'.format(logprob_mean, logprob_std)
    print(output)
    results_file = os.path.join(config.out_dir, 'results.txt')
    print(output, file=open(results_file, 'a'))
    return logprob_mean, logprob_std

def get_model(config):
    return MAF(config.model.n_blocks, config.model.input_size, config.model.hidden_size, config.model.n_hidden, None, 
                config.model.activation_fn, config.model.input_order, batch_norm=not config.model.no_batch_norm)
    
def train(args, config, dataloaders, device):
    train_dataloader, val_dataloader, test_dataloader = dataloaders 
    
    model = get_model(config)
          
    model = model.to(device)

    optimizer = get_optimizer(config, model.parameters())

    start_epoch, step = 0, 0
    best_eval_logprob = float('-inf')
    if args.resume_training:
        print('restoring checkpoint from {}'.format(args.restore_file))
        state = torch.load(os.path.join(args.restore_file, "best_model_checkpoint.pt"), map_location=device)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        start_epoch = state['epoch'] + 1
    model = torch.nn.DataParallel(model)

    # Training loop
    for epoch in range(start_epoch, config.training.n_epochs):
        data_start = time.time()
        data_time = 0
        # original maf code
        for i, data in enumerate(train_dataloader):
            
            
            # Sets model in training mode
            model.train()
            step += 1

            # check if labeled dataset
            if len(data) == 1:
                x, y = data[0], None
            else:
                x, y = data
                y = y.to(device)
            x = x.view(x.shape[0], -1).to(device)
            
            # Evaluate loss
            loss = -model.module.log_prob(x, y=None).mean(0)
            
            # Compute gradients
            loss.backward()
            optimizer.step()
            
            # Zero gradients
            optimizer.zero_grad()

            if i % config.training.log_interval == 0:
                print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch, start_epoch + config.training.n_epochs, i, len(train_dataloader), loss.item()))

            data_start = time.time()

        # now evaluate and save metrics/checkpoints
        eval_logprob, _ = test(
            model, test_dataloader, epoch, args, config, device)

        torch.save({
            'epoch': epoch,
            'model_state': model.module.state_dict(),
            'optimizer_state': optimizer.state_dict()},
            os.path.join(config.out_dir, 'model_checkpoint.pt'))
        # save model only
        torch.save(
            model.state_dict(), os.path.join(
                config.out_dir, 'model_state.pt'))
        # save best state
        if eval_logprob > best_eval_logprob:
            best_eval_logprob = eval_logprob
            print('saving model at epoch {}'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer_state': optimizer.state_dict()},
                os.path.join(config.out_dir, 'best_model_checkpoint.pt'))


class Gaussian(Dataset):
    def __init__(self, args, config, typ, split='train'):
        self.args = args
        self.config = config
        self.split = split
        self.type = typ
        assert self.type in ['bias', 'ref']

        self.p_mu = self.config.data.mus[0]
        self.q_mu = self.config.data.mus[1]

        self.perc = config.data.perc
        self.input_size = config.data.input_size
        self.label_size = 1
        self.base_dist = Normal(self.config.data.mus[0], 1)  # bias
        
        ### ADD HOTFIX
        try:
            fpath = os.path.join(self.config.training.data_dir, 'gmm')
            data = np.load(os.path.join(fpath, 'gmm_p{}_q{}.npz'.format(self.p_mu, self.q_mu)))
        except FileNotFoundError:
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            data = self.generate_data()

        if self.type == 'bias':
            data = data['p']
        else:
            data = data['q']

        # train/val/test split
        if split == 'train':
            data = data[0:40000]
        elif split == 'val':
            data = data[40000:45000]
        else:
            data = data[45000:]
        if self.type == 'ref' and self.split != 'val':  # keep val same
            to_keep = int(len(data) * self.perc)
            data = data[0:to_keep]
        self.data = torch.from_numpy(data).float()

    def generate_data(self):
        p = np.random.randn(50000,2) + self.p_mu
        q = np.random.randn(50000,2) + self.q_mu
        np.savez(os.path.join(self.config.training.data_dir, 'gmm', 'gmm_p{}_q{}.npz'.format(self.p_mu, self.q_mu)), **{
            'p': p,
            'q': q
        })
        return {'p': p, 'q': q}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        if self.type == 'bias':
            label = torch.zeros(1)
        else:
            label = torch.ones(1)

        return item, label

class PolicyData(Dataset):

    def __init__(self, data, typ, split):

        combined_action_context = data['AC']#transform(data['A'], data['C'])
        N = combined_action_context.shape[0]

        train_prop = .7
        val_prop = .15
        self.type = typ

        if split == "train":
            print(0, (int(train_prop * N)))
            data = combined_action_context[0: (int(train_prop * N) ),:]
        elif split == "val":
            data = combined_action_context[(int(train_prop * N)): (int( ( train_prop + val_prop) * N)), :]
        elif split == "test":
            data = combined_action_context[int( ( train_prop + val_prop) * N):, :]

        self.data = torch.from_numpy(data).float()
        print(split)
        print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        if self.type == "logging":
            label = torch.zeros(1)
        else:
            label = torch.ones(1)

        return item, label



def get_ordinary_dataloaders(args, config, device, data):
    input_dims = data['logging']['AC'].shape[1]
    label_size = 1
    lam = 1e-6
    batch_size = config.training.batch_size

    train_logging = PolicyData(data=data['logging'], typ='logging', split='train')
    val_logging = PolicyData(data=data['logging'], typ='logging', split='val')
    test_logging = PolicyData(data=data['logging'], typ='logging', split='test')

    train_target = PolicyData(data=data['target'], typ='target', split='train')
    val_target = PolicyData(data=data['target'], typ='target', split='val')
    test_target = PolicyData(data=data['target'], typ='target', split='test')

    train_dataset = ConcatDataset([train_logging, train_target])
    val_dataset = ConcatDataset([val_logging, val_target])
    test_dataset = ConcatDataset([test_logging, test_target])

    train_dataset.input_dims = input_dims
    train_dataset.input_size = int(np.prod(input_dims))
    train_dataset.label_size = label_size
    train_dataset.lam = lam

    test_dataset.input_dims = input_dims
    test_dataset.input_size = int(np.prod(input_dims))
    test_dataset.label_size = label_size
    test_dataset.lam = lam

    if val_dataset is not None:
        val_dataset.input_dims = input_dims
        val_dataset.input_size = int(np.prod(input_dims))
        val_dataset.label_size = label_size
        val_dataset.lam = lam

    # construct dataloaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == 'cuda' else {}
    print(len(train_dataset))
    print(train_logging.data.shape)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, **kwargs) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader

def get_gaussian_dataloaders(args, config, device):
    
    print("Getting dataloaders")
    input_dims = 2
    label_size = 1
    lam = 1e-6
    batch_size = config.training.batch_size
    
    train_biased = Gaussian(args, config, 'bias', split='train')
    val_biased = Gaussian(args, config, 'bias', split='val')
    test_biased = Gaussian(args, config, 'bias', split='test')

    train_ref = Gaussian(args, config, 'ref', split='train')
    val_ref = Gaussian(args, config, 'ref', split='val')
    test_ref = Gaussian(args, config, 'ref', split='test')

    if args.encode_z:
        # keep each dataset separate for encoding
        for dataset in (train_biased, train_ref):
            dataset.input_dims = input_dims
            dataset.input_size = int(np.prod(input_dims))
            dataset.label_size = label_size

        kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == 'cuda' else {}

        train_loader_biased = DataLoader(train_biased, batch_size, shuffle=True, **kwargs)
        val_loader_biased = DataLoader(val_biased, batch_size, shuffle=False, **kwargs)
        test_loader_biased = DataLoader(test_biased, batch_size, shuffle=False, **kwargs)
        train_loader_ref = DataLoader(train_ref, batch_size, shuffle=True, **kwargs)
        val_loader_ref = DataLoader(val_ref, batch_size, shuffle=False, **kwargs)
        test_loader_ref = DataLoader(test_ref, batch_size, shuffle=False, **kwargs)

        return [train_loader_biased, train_loader_ref], [val_loader_biased, val_loader_ref], [test_loader_biased, test_loader_ref]

    train_dataset = ConcatDataset([train_biased, train_ref])
    val_dataset = ConcatDataset([val_biased, val_ref])
    test_dataset = ConcatDataset([test_biased, test_ref])
    
    train_dataset.input_dims = input_dims
    train_dataset.input_size = int(np.prod(input_dims))
    train_dataset.label_size = label_size
    train_dataset.lam = lam

    test_dataset.input_dims = input_dims
    test_dataset.input_size = int(np.prod(input_dims))
    test_dataset.label_size = label_size
    test_dataset.lam = lam

    if val_dataset is not None:
        val_dataset.input_dims = input_dims
        val_dataset.input_size = int(np.prod(input_dims))
        val_dataset.label_size = label_size
        val_dataset.lam = lam

    # construct dataloaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == 'cuda' else {}

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, **kwargs) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, **kwargs)
    
    return train_loader, val_loader, test_loader


## Process args and configs

def set_up_args_and_configs(args):
    """
    Simulates args passed through from command line as dict.

    """
    args = dict2namespace(args)
    with open(os.path.abspath(args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    new_config.out_dir = os.path.join(new_config.training.out_dir, new_config.training.exp_id)
    print(os.path.exists(new_config.out_dir))
    if not os.path.exists(new_config.out_dir):
        print(new_config.out_dir)
        os.makedirs(new_config.out_dir, exist_ok=True)
    args.log_path = os.path.join(new_config.out_dir, 'logs')
    os.makedirs(new_config.out_dir, exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    return args, new_config

## Data generating processes
def generate_data(**kwargs):
    args = kwargs
    C = np.random.binomial(n=args['C_cardinality'] - 1, size=(args['n'], 1), p=.5)

    A_logging = C * np.random.multivariate_normal(mean=np.array([.5, .5]),
                                                  cov=args['A_scale'] * np.array([[1, .8], [.8, 1]]), size=args['n']) + \
                (1 - C) * np.random.multivariate_normal(mean=np.array([1, 1]),
                                                        cov=args['A_scale'] * np.array([[1, .8], [.8, 1]]),
                                                        size=args['n'])

    Y_logging = np.random.normal(loc=A_logging @ args['Y_coef'] + C.reshape(-1), scale=args['Y_scale'])
    A_target = C * np.random.multivariate_normal(mean=np.array([0, 0]),
                                                 cov=args['A_scale'] * np.array([[1, .8], [.8, 1]]), size=args['n']) + \
               (1 - C) * np.random.multivariate_normal(mean=np.array([.5, .5]),
                                                       cov=args['A_scale'] * np.array([[1, .8], [.8, 1]]),
                                                       size=args['n'])
    Y_target = np.random.normal(loc=A_target @ args['Y_coef'] + C.reshape(-1), scale=args['Y_scale'])

    AC_logging = get_kronecker_prod(A_logging, C)
    AC_target = get_kronecker_prod(A_target, C)
    data = {"logging":{"C": C, "A": A_logging, "Y": Y_logging, "AC": AC_logging}, "target": {"C": C, "A": A_target, "AC":AC_target, "Y": Y_target}}

    return data


def generate_embedded_AC_data(**kwargs):
    args = kwargs

    C_latent = np.random.binomial(n=1, size=(args['n']), p=.5)
    C = C_latent * np.random.normal(loc=-.5, scale=1, size=args['n']) + (1-C_latent) * np.random.normal(loc=.5, scale=1, size=args['n'])
    print(C.shape)
    A_logging_means = (np.array([[.2], [-.2]]) + C ).T
    print(A_logging_means.shape)
    A_logging = np.vstack([np.random.multivariate_normal(mean=A_logging_means[i, :],
                                                  cov=args['A_scale'] * np.array([[1, .3], [.3, 1]]), size=1) for i in range(A_logging_means.shape[0])])
    print(A_logging.shape)

    A_target_means = (np.array([[0], [0]]) + C ).T
    A_target = np.vstack([np.random.multivariate_normal(mean=x,
                                                 cov=args['A_scale'] * np.array([[1, .3], [.3, 1]]), size=1) for x in A_target_means])
    Y_logging = np.random.normal(loc=A_logging @ args['Y_coef'] + C.reshape(-1), scale=args['Y_scale'])

    Y_target = np.random.normal(loc=A_target @ args['Y_coef'] + C.reshape(-1), scale=args['Y_scale'])
    AC_logging = get_kronecker_prod(A_logging, C.reshape(-1, 1))
    AC_target = get_kronecker_prod(A_target, C.reshape(-1, 1))
    data = {"logging":{"C": C, "A": A_logging, "Y": Y_logging, "AC": AC_logging}, "target": {"C": C, "A": A_target, "AC":AC_target, "Y": Y_target}}

    return data

def generate_modified_kang_schafer_data(**kwargs):

    args = kwargs

    C = np.random.multivariate_normal(mean=np.zeros(4), cov=np.eye(4), size=args['n'])

    beta_1 = np.array([[1, -.5, .25, .1],
                       [.4, .3, .24, .2],
                       [.3, .2, .3, .1],
                       [-2, 1, .5, .3]])
    beta_2 = np.array([[.1, -.5, -.25, -.1],
                       [-.4, .8, -.34, .2],
                       [-.8, .9, -.3, 1],
                       [-2, 1, -.2, 3]])
    A_logging = C @ beta_1 + np.random.multivariate_normal(mean=np.zeros(4), cov=np.eye(4), size=args['n'])
    A_target =  C @ beta_2 + np.random.multivariate_normal(mean=np.zeros(4), cov=np.eye(4), size=args['n'])

    gamma_1 = np.array([27.4, 13.7, 13.7, 13.7])
    Y_logging = 210 + 10 * np.sum((A_logging), axis=1).reshape(-1) + (C @ gamma_1).reshape(-1) + np.random.normal(size=args['n'])
    Y_target = 210 + 8 * np.sum((A_target), axis=1).reshape(-1) + (C @ gamma_1).reshape(-1) + np.random.normal(size=args['n'])

    AC_logging = get_kronecker_prod(A_logging, C)
    AC_target = get_kronecker_prod(A_target, C)
    data = {"logging":{"C": C, "A": A_logging, "Y": Y_logging, "AC": AC_logging}, "target": {"C": C, "A": A_target, "AC":AC_target, "Y": Y_target}}

    return data


## Estimators

def get_kronecker_prod(A, C):
    i, j = C.shape
    i, l = A.shape
    AC = np.einsum("ij,il->ijl", C, A).reshape(i, j * l)
    return AC

# Compute PW estimator
def mult_rbf_kern(u, sig):
    """
    requires input of size n x p
    """
    return np.exp(- np.sum(u * u, axis=1) / sig)


def bandwidth_median_distance(x):
    """
    Selects variance using median distance heuristic. See https://arxiv.org/pdf/1707.07269.pdf
    """

    # return np.median(np.sqrt(np.sum(x * x, axis=1) / 2))
    dist_mat = (pairwise_distances(x))
    dist = dist_mat[np.triu_indices(dist_mat.shape[0], k=1)]
    med = np.median(dist)
    return med

def permutation_estimator(data):
    n = data["logging"]["A"].shape[0]

    features_logging = data["logging"]["AC"]
    features_target = data["target"]["AC"]
    Z = [0] * n + [1] * n

    features = np.concatenate([features_logging, features_target], axis=0)
    clf = GradientBoostingClassifier(n_estimators=50, min_samples_leaf=2)
    clf = clf.fit(X=features, y=Z)
    prob = clf.predict_proba(X=features_logging)[:, 1]
    density_ratio = prob / (1 - prob)

    diffs = features_logging - features_target
    bwh = bandwidth_median_distance(diffs)
    kerns = mult_rbf_kern(diffs, bwh)
    weights = density_ratio * kerns

    policy_value = np.mean(data["logging"]["Y"] * weights) / np.mean(weights)

    result = {"result": policy_value,
              "density_ratio": density_ratio,
              "kerns": kerns,
              "weights": weights}

    return result

def get_transformed_data(model, data):
    """
    Transforms the AC-matrix using the featurized normalizing flow model
    Does not transform the A or C matrices!

    Args:
        model:
        data:

    Returns:

    """

    transformed_data = copy.deepcopy(data)
    for policy_type in ["logging", "target"]:
        transformed_actions, _ = model.forward(torch.from_numpy(data[policy_type]["AC"]).float().to(device))
        transformed_actions = transformed_actions.detach().cpu().numpy() # convert to numpy array
        transformed_data[policy_type]["AC"] = transformed_actions
    return transformed_data

def get_bootstrapped_data(data, ix):
    new_data = copy.deepcopy(data)
    def _get_bootstrapped_data(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = _get_bootstrapped_data(v)
            else:
                d[k] = v[ix]
        return d

    return _get_bootstrapped_data(new_data)

if __name__ == "__main__":

    # Set up args and devices
    args_dict = {'config': '../src/configs/flows/gmm/maf_copy.yaml',
            'resume_training': False,
            'restore_file': False,
            'seed': 1234,
            'encode_z': False}
    args, config = set_up_args_and_configs(args=args_dict)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Generate simulation data:
    # In this step we create data from simulated logging and target policies.

    # Specify data simulation params
    params = {'n': 5000,
              'Y_coef': np.array([1, 1]),
              'A_scale': .01,
              'Y_scale': .01,
              'embedding_dim': 1,
              'C_cardinality': 2,
              'eps': 10
              }

    data = generate_modified_kang_schafer_data(**params)


    # Train Normalizing Flow Model
    ## Load data to train
    if not config.training.use_cached:
        dataloaders = get_ordinary_dataloaders(args, config, device, data)

        ## Train model
        train(args, config, dataloaders, device)

    # Fit density ratio model


    ## Use normalizing flow model to transform actions
    model = get_model(config)
    restore_file = os.path.abspath('../src/flows/results/{}/'.format(config.training.exp_id))
    state = torch.load(os.path.join(restore_file, "best_model_checkpoint.pt"), map_location='cpu')
    model.load_state_dict(state['model_state'])
    model = model.to(device)
    model.eval()

    # Get MAF transformed actions

    # Compute BOPE estimator
    data_list = []
    for i in range(config.estimation.bootstraps):
        ix = np.random.choice(a=params['n'], size=params['n'])
        temp_data = get_bootstrapped_data(data, ix)
        transformed_temp_data = get_transformed_data(model, temp_data)
        pw_estimate = permutation_estimator(temp_data)["result"]
        f_pw_estimate = permutation_estimator(transformed_temp_data)["result"]
        target = np.mean(temp_data["target"]["Y"])
        logging = np.mean(temp_data["logging"]["Y"])

        results = {"pw_estimate": pw_estimate, "f_pw_estimate": f_pw_estimate,
                   "target": target, "logging": logging}
        data_list.append(results)

    results_df = pd.DataFrame(data_list)

    results_df.to_csv(os.path.join(os.path.abspath('../src/flows/results/{}/'.format(config.training.exp_id)), "estimates.csv"))





