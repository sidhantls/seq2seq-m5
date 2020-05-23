import os 
import core, dataset 
import torch
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss, PoissonNLLLoss
import torch

if __name__ == '__main__':
    data_dir = Path('data/dataset4')
    normalize_using_category = 'id'
    # scaling factors after differencing
    with open(data_dir/"scale_factors_dict_diff.pkl", 'rb') as f:
        norm_factors=pickle.load(f)
    norm_factors['mean'] = torch.Tensor(norm_factors['mean']).float()
    norm_factors['std'] = torch.Tensor(norm_factors['std']).float()
    # parsing categorical feature names and creating index feature map
    with open(data_dir/'feature_names.txt', 'r') as f:
        feature_names = f.read()
    feature_names=feature_names.split(',')
    categorical_details = dataset.categorical_details.copy()
    maps = dataset.get_mappings(feature_names, cont_features=None, cat_features=categorical_details.keys())
    norm_idx = maps['cat'][normalize_using_category]
    for cat in dataset.ignore_categories: 
        categorical_details.pop(cat)
        maps['cat'].pop(cat)

    # create loader
    normalizer = core.Normalizer(norm_factors=norm_factors, norm_using_idx=15, norm_idx=0)
    train_dataset, valid_dataset = dataset.create_datasets_stratified_presplit(data_dir, 
                                                            norm_vec=norm_factors['mean'], 
                                                            prob_divides=[0.55],
                                                            prob_divides_val=[0.55])
    print('Length of validation dataset ', len(valid_dataset))
    print('Train Samples: {} Validation samples: {}'.format(len(train_dataset), len(valid_dataset)))

    batch_size=128*4
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                shuffle=False, pin_memory=True, num_workers=0)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                shuffle=False,pin_memory=True, num_workers=0)

    model=core.Seq2SeqRNNAttn(maps, categorical_details, nl=2, nh=128).cuda()
    # compiling with torch script
    # model = torch.jit.script(model).cuda()
    assert train_dataset[0][0].shape[0] != (model.dec_seq_len+1)
    # criterion = MSELoss()
    criterion = PoissonNLLLoss(log_input=False)
    optimizer = Adam(model.parameters())

    epochs= 10
    scheduler= torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, steps_per_epoch=len(train_dataloader), 
                                                epochs=epochs, div_factor=10)
    comments = 'Attention, Differencing before Normalization, train-val sampling, re-constr before loss, Poisson Loss'
    print('Starting Training..')
    model = core.train(model, criterion, optimizer, train_dataloader, valid_dataloader, nb_epochs=epochs,
                    comments=comments,
                    scheduler=scheduler, Normalizer=normalizer)
    torch.save(model.state_dict(), f'models/{comments}.pth')
