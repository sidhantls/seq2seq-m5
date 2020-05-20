import os 
import core, dataset 
import torch
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_dir = Path('data/dataset')
    normalize_using_category = 'id'
    with open(data_dir/'feature_names.txt', 'r') as f:
        feature_names = f.read()
    norm_vec = torch.from_numpy(np.load(data_dir/'item_scale_factor.npy').astype(np.float32))

    feature_names=feature_names.split(',')
    categorical_details = dataset.categorical_details.copy()
    maps = dataset.get_mappings(feature_names, cont_features=None, cat_features=categorical_details.keys())

    norm_idx = maps['cat'][normalize_using_category]

    for cat in dataset.ignore_categories:
        maps['cat'].pop(cat)
        categorical_details.pop(cat)

    length = 1006170
    # train_idx, valid_idx = core.create_train_test_split(length)
    # train_dataset = core.dataset_sampling(data_dir, norm_vec, train=True, idx_split=train_idx)
    # valid_dataset = core.dataset_sampling(data_dir, norm_vec, train=False, idx_split=valid_idx)

    # train_dataset, valid_dataset = dataset.create_datasets_stratified_(data_dir, norm_vec, prob_divides=[0.5, 0.8], prob_divides_val=[0.80])
    normalizer = core.Normalizer(norm_factors=norm_vec, norm_using_idx=15, norm_idx=0)
    train_dataset, valid_dataset = dataset.create_datasets_stratified_presplit(data_dir, 
                                                            norm_vec, 
                                                            prob_divides=[0.65],
                                                            prob_divides_val=[0.65])

    batch_size=128*4
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                shuffle=False, pin_memory=True, num_workers=0)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,shuffle=False,pin_memory=True, num_workers=0)


    from torch.optim import Adam
    from torch.nn import MSELoss, L1Loss, PoissonNLLLoss
    import torch


    model=core.Seq2SeqRNN(maps, categorical_details, nl=2, nh=256).cuda()
    # compiling with torch script
    # model = torch.jit.script(model).cuda()

    criterion = MSELoss()
    # criterion = PoissonNLLLoss(log_input=False)

    optimizer = Adam(model.parameters())

    epochs= 35
    scheduler= torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, steps_per_epoch=len(train_dataloader), 
                                                epochs=epochs, div_factor=10)

    comments = 'MSE Loss, FC at global, all local sampling, new train test split, all local'

    normalizer = core.Normalizer(norm_factors=norm_vec, norm_using_idx=15, norm_idx=0)
    print('Starting Training..')
    model = core.train(model, criterion, optimizer, train_dataloader, valid_dataloader, nb_epochs=epochs,
                    comments=comments,
                    scheduler=scheduler, Normalizer=normalizer)
    torch.save(model.state_dict(), f'models/{comments}.pth')

    
