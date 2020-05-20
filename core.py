from torch import nn
import torch
from collections import OrderedDict
import h5py
import os
from pathlib import Path
import numpy as np
from torch.nn import MSELoss
import matplotlib.pyplot as plt


def create_cat_inputs_embeddings(categorical_details):
    """
    Creates cat embedding layers given categorical details dict
    """
    embedding_layers={}
    for feature_name, embedding_params in categorical_details.items():
        embedding_layer = nn.Embedding(
            num_embeddings = categorical_details[feature_name]['input_dim'],
            embedding_dim = categorical_details[feature_name]['output_dim']
        )
        embedding_layers[feature_name] = embedding_layer
    return embedding_layers

class Seq2SeqRNN(nn.Module):
    def __init__(self, maps, categorical_details,
                    nh=128, nl=2, pr_force=0, implementation_1=True, device='cuda'):
        super().__init__()
        """
        maps {cont: cat: ) mappings to indices
        """
        self.nl,self.nh = nl,nh
        # embedding input size encoder
        self.cont_size = len(maps['cont'])
        # num global dim
        global_local_size = [0, 0]
        for key, val in categorical_details.items():
            if val['global']:
                global_local_size[0]+=val['output_dim']
            else:
                global_local_size[1]+=val['output_dim']
        print('global local sizes ', global_local_size)
        self.em_sz_dec = nh
        self.em_sz_enc = self.cont_size + global_local_size[1]
        # maps are dicts of mapings of cont/cat features with their index in input x
        self.cat_map = maps['cat']
        self.cont_map = maps['cont']
        self.cont_idx = list(maps['cont'].values())
        # embedding layers
        self.categorical_details = categorical_details
        self.embedding_layers = nn.ModuleDict(create_cat_inputs_embeddings(categorical_details))
        # encoder - input layer
        self.gru_enc_n = nn.GRU(self.em_sz_enc, nh, num_layers=nl,
                      dropout=0., batch_first=True, bidirectional=False)
        self.fc_global = nn.Linear(nh + global_local_size[0], self.nh)
        # decoder input layer
        self.gru_dec_1 = nn.GRUCell(1, self.em_sz_dec)
        # nth layer, n>1
        self.gru_dec_n = nn.GRUCell(self.em_sz_dec, self.em_sz_dec)
        # fc
        self.fc = nn.Linear(self.em_sz_dec, 1)
        self.out_act = nn.ReLU()
        self.pr_force = pr_force
        self.dec_seq_len = 28

    def calculate_embeddings(self, inp):
        """
        Uses categorical mappings to get indices of respective features of input to pass embedding layers
        Returns global and local emeddings based on definition in categorical map
        """
        embeddings_global = []
        embeddings_local = []
        for key, cat_idx in self.cat_map.items():
            if key not in self.embedding_layers.keys():
                raise ValueError('Categorical name {} in cat map not in categorical details'.format(key))
            if self.categorical_details[key]['global']:
                embeddings_global.append(self.embedding_layers[key](inp[:, 0, cat_idx].long())) # all timesteps are eq of global features
            else:
                embeddings_local.append(self.embedding_layers[key](inp[:, :, cat_idx].long()))

        if embeddings_global: embeddings_global = torch.cat(embeddings_global, dim=-1)
        if embeddings_local: embeddings_local = torch.cat(embeddings_local, dim=-1)
        #if embeddings_local.shape[-1] != (self.em_sz_enc-self.cont_size): raise ValueError('Final embedding size mismatch')
        return embeddings_global, embeddings_local

    def encoder(self, inp):
        """
        Input: features input
        Output: Output encoder hidden state- if global features are defined, their embeddings are
        concatenated to this output hidden state
        """

        # get embeddings for categoricals
        embeddings_global, embeddings_local = self.calculate_embeddings(inp)
        # combine embeddings with continous features
        if len(embeddings_local) != 0:
            inp = torch.cat([inp[:, :, self.cont_idx], embeddings_local], axis=-1)
        else:
            inp=inp[:, :, self.cont_idx]

        if inp.shape[-1] != self.em_sz_enc:
            raise ValueError('missmatch in encoder input dim {} and actual input {}'.format(inp.shape[-1], self.em_sz_enc))

        o, h = self.gru_enc_n(inp)
        h = h[-1, :, :] # last hidden out

        # global conditioning
        if len(embeddings_global) != 0:
            h = torch.cat([h, embeddings_global], axis=1)
            h = self.fc_global(h)

        assert h.shape[-1] == self.em_sz_dec

        return h
    def decoder(self, dec_inp, h):
        """
        Decoder at t=0
        Inputs: dec_inp: (batch_size, dec_input_size)-> dec_input_size=1
                h: encoder output (batch_size, hidden_size)
        Outputs: Decoder output at t=0 and hidden states
        """
        hiddens = []
        for i in range(self.nl):
            if i == 0:
                o  = self.gru_dec_1(dec_inp, h)
            else:
                o = self.gru_dec_n(o) # hidden of nth layer of 1st timestep initialized to 0's
            hiddens.append(o)

        outp = self.out_act(self.fc(o))
        assert len(outp.shape) == 2 and outp.shape[-1] == 1
        return outp, hiddens
    def decode_n(self, dec_inp, hiddens):
        # build layers
        for i in range(self.nl):
            if i == 0:
                # hidden from previous decoder
                o  = self.gru_dec_1(dec_inp,hiddens[i])
                hiddens[i] = o
            else:
                o = self.gru_dec_n(o, hiddens[i])
                hiddens[i] = o

        outp = self.out_act(self.fc(o))
        return outp, hiddens
    def forward(self, inp, targ=None):
        h = self.encoder(inp)
        # passing in last input of encoder
        dec_inp = inp[:, -1, 0].unsqueeze(-1)
        assert dec_inp.shape[-1] == 1
        res = []
        for i in range(self.dec_seq_len):
            if i == 0:
                outp, hiddens = self.decoder(dec_inp, h)
            else:
                outp, hiddens = self.decode_n(dec_inp,hiddens)
            res.append(outp)
            dec_inp = outp

            if (targ is not None) and (random.random()<self.pr_force):
                if i==self.dec_seq_len-1: continue
                dec_inp = targ[:,i]
                assert len(dec_inp.shape) == 2

        preds = torch.cat(res, axis=1).unsqueeze(-1)
        assert preds.shape[2] == 1 and len(preds.shape) == 3

        return preds
    def initHidden(self, bs): return torch.zeros((self.nl, bs, self.nh))


def predict_test(model, test_dataloader, cont_map, norm_factors, idx=14, cuda=True):
    model.eval()
    preds=[]
    model.pr_force=0

    for x in test_dataloader:
        x, norm_vec=normalize(x, norm_idx=0, norm_using_idx=idx,
                              norm_factors=norm_factors, remove=True, target='sales')
        if cuda:
            x = x.cuda()
            norm_vec = norm_vec.cuda()

        with torch.no_grad():
            pred = model(x)
            pred[:, :, 0] = pred[:, :, 0] * norm_vec

        preds.append(pred.cpu())

    preds = torch.cat(preds, axis=0)
    return preds


from torch.utils.tensorboard import SummaryWriter
from time import time
import random
from torch.functional import F
import subprocess
import os

def normalize(x, norm_idx, norm_using_idx, norm_factors, remove=True, target='sales'):
    # normalize index 0
    cat_ids = x[:, 0, norm_using_idx].long()
    norm_vec = norm_factors[cat_ids].unsqueeze(-1)
    x[:, :, norm_idx] = x[:, :, norm_idx]/norm_vec
    return x , norm_vec

class Normalizer(object):
    def __init__(self, norm_factors, norm_using_idx=15, norm_idx=0):
        self.norm_factors = norm_factors
        self.norm_using_idx = norm_using_idx
        self.norm_idx = norm_idx
    def get_norm_vec(self, x):
        cat_ids = x[:, 0, self.norm_using_idx].long()
        norm_vec = self.norm_factors[cat_ids].unsqueeze(-1)
        return norm_vec
    def normalize(self, x):
        norm_vec = self.get_norm_vec(x)
        x[:, :, self.norm_idx] = x[:, :, self.norm_idx]/norm_vec
        return x, norm_vec


def eval_model(model, valid_dataloader, criterion, Normalizer):
    model.eval()
    losses = []
    scores_list = []
    for x, target in valid_dataloader:
        x, norm_vec = Normalizer.normalize(x)
        # normalize
        target[:, :, 0] = target[:, :, 0]/norm_vec
        target = target.cuda()
        norm_vec = norm_vec.cuda()
        x = x.cuda()
        # forward + backward + optimize
        with torch.no_grad():
            outputs = model(x, target)
            # unnormalize
            outputs[:, :, 0] = outputs[:, :, 0]*norm_vec
            target[:, :, 0] = target[:, :, 0]*norm_vec
            loss = criterion(outputs, target)
            losses.append(loss.item())
            scores_list.append(eval_metrics(outputs, target))
    result = {}
    for key in scores_list[0]:
        value = sum(item[key] for item in scores_list)/len(scores_list)
        result[key] = value
    result['Loss'] =sum(losses)/len(losses)

    return result

def eval_metrics(pred, target):
    rmse = torch.sqrt(F.mse_loss(pred, target))
    return {'rmse': rmse}


def train(model, criterion, optimizer, train_dataloader, valid_dataloader, comments='', nb_epochs=5,
          scheduler=None, pr_force=True, Normalizer=None, last_epoch=0):

    writer = SummaryWriter(comment=comments, max_queue=0)

    for epoch in range(nb_epochs):
        epoch=last_epoch+epoch
        running_loss = 0.0
        epoch_loss=0
        scores_list = []

        model.train()
        epoch_start = time()

        if pr_force:
            model.pr_force = np.clip(1 - epoch/(nb_epochs-3), 0, 1)
            writer.add_scalar('train_params/pr_force', model.pr_force, epoch)

        for i, data in enumerate(train_dataloader):
            n_iter = len(train_dataloader)*epoch+i
            # get the inputs; data is a list of [inputs, labels]
            x, target = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            x, norm_vec = Normalizer.normalize(x)
            x=x.cuda()
            target=target.cuda()
            norm_vec=norm_vec.cuda()
            t_force=target.clone()
            # normalization
            t_force[:, :, 0] = t_force[:, :, 0]/norm_vec
            outputs = model(x, t_force)
            # reverse the normalization
            outputs[:, :, 0] = outputs[:, :, 0]*norm_vec
            # loss
            loss = criterion(outputs, target)
            loss.backward()

            if scheduler != None:
                optimizer.step()
                scheduler.step()
                if i%100==0:
                    last_lr = scheduler.get_last_lr()
                    writer.add_scalar('train_params/LR',last_lr[0], n_iter)
            else:
                optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss+=running_loss
            i+=1
            scores_list.append(eval_metrics(outputs, target))

            if i % 500 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i, running_loss / 500))
                n_step = len(train_dataloader)*epoch+i
                writer.add_scalar('Loss/train', running_loss / 500, n_iter)
                running_loss = 0.0
                for key in scores_list[0]:
                    value = sum(item[key] for item in scores_list)/len(scores_list)
                    writer.add_scalar(f'{key}/train', value, n_iter)
                scores_list = []

        epoch_loss=epoch_loss/len(train_dataloader)
        print('Finished Training Epoch, validating.. ')
        print('Epoch trained in {%.3f} seconds' % (time()-epoch_start))
        metrics= eval_model(model, valid_dataloader, criterion,
                   Normalizer=Normalizer)

        for key, value in metrics.items(): writer.add_scalar(f'{key}/valid', value, epoch)

        print('Epoch train loss: {}, epoch valid loss {}'.format(epoch_loss, metrics['Loss']))
        print('Epoch trained and valided in {%.3f} seconds' % (time()-epoch_start))
        model_dir = f'models/{comments}'

        if os.path.isdir(model_dir):
            torch.save(model.state_dict(), f'{model_dir}/{epoch}.pth')
        else:
            os.mkdir(model_dir)
            torch.save(model.state_dict(), f'{model_dir}/{epoch}.pth')

        epoch_loss = 0

    return model