from torch import nn
import torch
from collections import OrderedDict
import h5py
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random

categorical_details = {
        'event_name_1': {'input_dim': 31, 'output_dim': 1, 'global': False},
        'event_type_1': {'input_dim': 5, 'output_dim': 1, 'global': False},
        'event_name_2': {'input_dim': 5, 'output_dim': 1, 'global': False},
        'event_type_2': {'input_dim': 3, 'output_dim': 1, 'global': False},
        'item_id': {'input_dim': 3049, 'output_dim': 3, 'global': True},
        'dept_id': {'input_dim': 7, 'output_dim': 2, 'global': True},
        'store_id': {'input_dim': 10, 'output_dim': 3, 'global': True},
        'cat_id': {'input_dim': 3, 'output_dim': 1, 'global': True},
        'state_id': {'input_dim': 3, 'output_dim': 1, 'global': True},
        'snap_CA': {'input_dim': 2, 'output_dim': 1, 'global': False},
        'snap_WI': {'input_dim': 2, 'output_dim': 1, 'global': False},
        'snap_TX': {'input_dim': 2, 'output_dim': 1, 'global': False},
        'id': {'input_dim': 30490, 'output_dim': 3, 'global': False},
        # 'year': {'input_dim': 6, 'output_dim': 2, 'global': False},
        'is_valid':{'input_dim': None, 'output_dim': None, 'global': False}
}

ignore_categories = ['event_name_1', 'event_name_2', 'is_valid', 'id', 'snap_CA', 'snap_TX']

def get_mappings(features, cont_features, cat_features):
    """
    inputs:
        features - list of features
        cont_features - list. If none, everything other than Cat is used
        cat_features = categorical features in order of the embeddings in neural net
        target - json with level 0 of cat or cont keys and then level 2 of feature name-idx mapping
    """
    # auto
    if not cont_features:
        cont_features = list(set(features) - set(cat_features))

    feature_map = {'cont': {}, 'cat': {}}
    for i, feat in enumerate(features):
        if feat in cat_features:
            feature_map['cat'][feat] = i
        elif feat in cont_features:
            feature_map['cont'][feat] = i
    return feature_map

def train_sampler(dataloader, norm_vec, norm_using_idx):
    probs=[]
    for x, target in dataloader:
        ids = x[:, 0, norm_using_idx]
        norm = norm_vec[ids.long()]
        prob_sampling = 1-1/norm
        probs.append(prob_sampling)

    return probs

def get_dataset_probs(x, norm_vec, norm_using_idx):
    """
    Returns list containing probability of sampling each index from dataset
    """
    ids = x[:, 0, norm_using_idx]
    norm = norm_vec[ids.long()]
    prob_sampling = 1-1/norm
    return prob_sampling.tolist()

def find_hprob_start(nums, target=0.70):
    """
    Binary search to find index of max value less than target.
    """
    if not nums: return False
    left, right = 0, len(nums)-1
    while left < right:
        mid = (left+right)//2
        if nums[mid] <= target and nums[mid] >= target: return mid
        elif nums[mid] < target: left = mid+1
        elif nums[mid] > target: right=mid
    return mid

def create_train_test_split(length):
    np.random.seed(40)
    rng = np.random.default_rng()
    train_idx = rng.choice(length, size=length*85//100, replace=False)
    valid_idx = np.setdiff1d(list(range(length)), train_idx)
    return train_idx, valid_idx


def create_samples(probs: list, prob_divides = [0.5, 0.8]):
    """
    Input: Probabilties of each sample in dataset
    Output: List of indices in dataset with stratified uniform sampling
    """
    if not prob_divides:
        return list(range(0, len(probs)))

    probs=torch.Tensor(probs)
    sorted_idx = probs.sort().indices.tolist()
    sorted_probs = probs.sort().values.tolist()
    prob_idxs = []
    start_idx=0

    for thresh in prob_divides:
        end_idx = find_hprob_start(sorted_probs, thresh)
        prob_idxs.append(sorted_idx[start_idx:end_idx])
        start_idx = end_idx
    if end_idx != (len(sorted_idx)-1):
        prob_idxs.append(sorted_idx[end_idx:])

    for idx_list in prob_idxs:random.shuffle(idx_list)

    len_low_prob_items = len(prob_idxs[0])
    new_idxs = []
    for i in range(len_low_prob_items):
        for k, part_nb in enumerate(range(len(prob_idxs))):
            part = prob_idxs[part_nb]
            new_idxs.append(part[i%len(part)])
    return new_idxs

class dataset_sampling(torch.utils.data.Dataset):
    def __init__(self, data_dir, norm_vec, train=True, idx_split=[]):
        super().__init__()
        self.y = torch.tensor(np.load(data_dir/"y_train_48.npy"))[idx_split, :, :]
        self.x = np.load(data_dir/"x_train_48.npy")[idx_split, :, :]
        self.x = torch.from_numpy(self.x)
        self.train = train
        if train:
            probs = get_dataset_probs(self.x, norm_vec)
            self.sampling_idx = create_samples(probs)
        print('Features shape {}, target shape {}'.format(self.x.size(), self.y.size()))
    def __getitem__(self, idx):
        if self.train:
            return self.x[self.sampling_idx[idx], :, :],  self.y[self.sampling_idx[idx], :, :].float()
        else:
            return self.x[idx, :, :], self.y[idx, :, :].float()

    def __len__(self):
        if self.train:
            return len(self.sampling_idx)
        else:
            return len(self.y)


def create_datasets_stratified(data_dir, norm_vec, prob_divides, prob_divides_val):
    import random
    from sklearn.model_selection import StratifiedShuffleSplit
    class dataset_sampling_stratified(torch.utils.data.Dataset):
        def __init__(self, data_dir, norm_vec, train=True, valid_idx=[], series_idx=14, prob_divides = []):
            super().__init__()
            self.y = torch.tensor(np.load(data_dir/"y_train.npy"))
            self.x = np.load(data_dir/"x_train.npy")
            self.x = torch.from_numpy(self.x)
            self.train = train
            if train:
                StrafiedSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.15, train_size=0.85, random_state=42)
                self.train_idx, self.valid_idx = list(StrafiedSplit.split(
                    X=np.zeros(len(self.x)), y=self.x[:, 0, series_idx]))[0]
                self.x = self.x[self.train_idx]
                self.y = self.y[self.train_idx]
                probs = get_dataset_probs(self.x, norm_vec)
                self.sampling_idx = create_samples(probs, prob_divides = prob_divides)
            else:
                assert len(valid_idx) !=0
                self.x = self.x[valid_idx]
                self.y = self.y[valid_idx]

                probs = get_dataset_probs(self.x, norm_vec)
                self.sampling_idx = create_samples(probs, prob_divides = prob_divides)
        def __getitem__(self, idx):
            return self.x[self.sampling_idx[idx], :, :],  self.y[self.sampling_idx[idx], :, :].float()

        def __len__(self):
            return len(self.sampling_idx)

    train_dataset = dataset_sampling_stratified(data_dir, norm_vec, train=True, prob_divides=prob_divides)
    valid_dataset = dataset_sampling_stratified(data_dir, norm_vec, train=False,
                                                valid_idx=train_dataset.valid_idx, prob_divides=prob_divides_val)
    return train_dataset, valid_dataset

def create_datasets_stratified_presplit(data_dir, norm_vec, prob_divides, prob_divides_val):
    import random
    from sklearn.model_selection import StratifiedShuffleSplit
    class dataset_sampling_stratified(torch.utils.data.Dataset):
        def __init__(self, data_dir, norm_vec, train=True, is_valid_idx=14, series_idx=15, prob_divides = []):
            super().__init__()
            self.y = torch.tensor(np.load(data_dir/"y_train.npy"))
            self.x = np.load(data_dir/"x_train.npy")
            is_valid = self.x[:, :, is_valid_idx].astype(np.bool)
            is_valid = is_valid.any(1)
            if is_valid.shape[0] != len(self.x) or len((is_valid.shape))>1:
                raise ValueError(is_valid.shape)
            if not train:
                self.x = self.x[is_valid, :, :]
                self.y = self.y[is_valid, :, :]
            else:
                self.x = self.x[~is_valid, :, :]
                self.y = self.y[~is_valid, :, :]
            self.x = torch.from_numpy(self.x)
            self.train = train
            probs = get_dataset_probs(self.x, norm_vec, norm_using_idx=series_idx)
            if train:
                self.sampling_idx = create_samples(probs, prob_divides = prob_divides)
            else:
                self.sampling_idx = create_samples(probs, prob_divides = prob_divides)

        def __getitem__(self, idx):
            return self.x[self.sampling_idx[idx], :, :],  self.y[self.sampling_idx[idx], :, :].float()

        def __len__(self):
            return len(self.sampling_idx)

    train_dataset = dataset_sampling_stratified(data_dir, norm_vec, train=True, prob_divides=prob_divides)
    valid_dataset = dataset_sampling_stratified(data_dir, norm_vec, train=False,
                                                prob_divides=prob_divides_val)
    print('train test split: ', len(train_dataset)/len(train_dataset))
    return train_dataset, valid_dataset

def create_datasets_stratified_presplit_onlyval(data_dir, norm_vec, prob_divides, prob_divides_val):
    import random
    from sklearn.model_selection import StratifiedShuffleSplit
    class dataset_sampling_stratified(torch.utils.data.Dataset):
        def __init__(self, data_dir, norm_vec, train=False, is_valid_idx=14, series_idx=15, prob_divides = []):
            super().__init__()
            self.y = torch.tensor(np.load(data_dir/"y_train.npy"))
            self.x = np.load(data_dir/"x_train.npy")
            is_valid = self.x[:, 0, is_valid_idx].astype(np.bool)
            if not train:
                self.x = self.x[is_valid, :, :]
                self.y = self.y[is_valid, :, :]
            else:
                self.x = self.x[~is_valid, :, :]
                self.y = self.y[~is_valid, :, :]
            self.x = torch.from_numpy(self.x)
            self.train = train

            probs = get_dataset_probs(self.x, norm_vec, norm_using_idx=series_idx)
            if train:
                self.sampling_idx = create_samples(probs, prob_divides = prob_divides)
            else:
                self.sampling_idx = create_samples(probs, prob_divides = prob_divides)

        def __getitem__(self, idx):
            return self.x[self.sampling_idx[idx], :, :],  self.y[self.sampling_idx[idx], :, :].float()

        def __len__(self):
            return len(self.sampling_idx)

    # train_dataset = dataset_sampling_stratified(data_dir, norm_vec, train=True, prob_divides=prob_divides)
    valid_dataset = dataset_sampling_stratified(data_dir, norm_vec, train=False,
                                                prob_divides=[])
    return valid_dataset

# for kaggle file, ignore 
def create_submissions(preds, data_dir):
    """preds of shape samples x seq_len"""
    import pandas as pd
    # preds is tensor of samples x pred_len
    df= pd.read_csv(data_dir/'sample_submission.csv')
    df.head()
    cont_cols = df.columns[1:]
    df.at[:(len(df)//2)-1, cont_cols] = np.array(preds)
    all_cols = df.columns.tolist()
    all_cols = [all_cols.pop(0)] + all_cols
    df = df[all_cols]
    return df

class dataset_test(torch.utils.data.Dataset):
    def __init__(self, data_dir, normalize_stats={}):
        super().__init__()
        self.x = np.load(data_dir/"x_test.npy").astype(np.float32)
        self.x = torch.from_numpy(self.x)
    def __getitem__(self, idx):
        return self.x[idx, :, :]
    def __len__(self):
        return len(self.x)