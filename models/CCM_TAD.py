# -*- coding: utf-8 -*-
"""
Created when I should've been asleep
@author: Murad
SISLab, USF
mmurad@usf.edu
"""

import torch
import torch.nn as nn
from utils.tools import Permute
from models.CausalLinearLayer import CausalLinearLayer
import numpy as np


class CCM_TAD(nn.Module):
    def __init__(self, args):
        
        super().__init__()
        self.args = args
        self.pred_len = self.args.pred_len
        self.channel_in = self.args.c_in
        self.seq_len = self.args.seq_len
        self.d_model = self.args.d_model
        self.dropout = self.args.dropout
        self.embedding_dropout = self.args.embedding_dropout
        self.input_dropout = self.args.input_dropout
        self.batch_size = self.args.batch_size 
        self.tfactor = self.args.tfactor
        self.dfactor = self.args.dfactor
        self.use_amp = self.args.use_amp
        self.device = self.args.device
        self.num_mix = self.args.num_mix
        self.nheads = self.args.nheads
        self.clusterid = self.args.clusterid
        mixerType = self.args.mixerType # 'anti_causal' # 'anti_causal', 'causal', 'linear'
        
        self.__model__ = __Model__(
            input_seq = self.seq_len,
            pred_seq = self.pred_len,
            batch_size = self.batch_size,
            channel = self.channel_in,
            d_model = self.d_model,
            dropout = self.dropout,
            embedding_dropout = self.embedding_dropout,
            input_dropout = self.input_dropout,
            tfactor = self.tfactor,
            dfactor = self.dfactor,
            num_mix = self.num_mix,
            nheads = self.nheads,
            clusterid = self.clusterid,
            mixerType = mixerType)

    def forward(self, x,):
        y = self.__model__(x)
        y = y[:, -self.pred_len:, :] # decomposition output is always even, but pred length can be odd
        return y
    


class __Model__(nn.Module):
    def __init__(self, 
                 input_seq = [], 
                 pred_seq = [], 
                 batch_size = [], 
                 channel = [], 
                 d_model = [], 
                 dropout = [], 
                 embedding_dropout = [], 
                 input_dropout = [],
                 tfactor = [], 
                 dfactor = [], 
                 num_mix = [], 
                 nheads = [], 
                 clusterid = [], 
                 mixerType = []):
        
        super().__init__()
        self.input_seq = input_seq
        self.pred_seq = pred_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.input_dropout = input_dropout
        self.tfactor = tfactor
        self.dfactor = dfactor
        
        self.num_mix = num_mix
        self.norm_type = 'batch'
        self.nheads = nheads
        self.clusterid = clusterid
        self.mixer_input = self.input_seq
        
        self.multi_embedding = MultiHeadEmbedding(d_model = self.d_model, 
                                                  channel = self.channel, 
                                                  nheads = self.nheads, 
                                                  clusterid = self.clusterid)
        
        self.mixers = nn.ModuleList([
            CausalMixer(input_seq = self.mixer_input, 
                        out_seq = self.mixer_input, 
                        batch_size = self.batch_size, 
                        channel = self.channel,
                        d_model = self.d_model,
                        dropout = self.dropout,
                        tfactor = self.tfactor, 
                        dfactor = self.dfactor,
                        mixerType = mixerType) for i in range(self.num_mix)])
        
        self.norm = nn.Sequential(Permute(0, -1, 1), 
                                  nn.BatchNorm1d(self.d_model), 
                                  Permute(0, -1, 1))
        
        self.norm0 = nn.Sequential(Permute(0, -1, 1), 
                                   nn.BatchNorm1d(self.d_model), 
                                   Permute(0, -1, 1))
        
        self.dropoutLayer = nn.Dropout(self.embedding_dropout) 
        self.dropoutLayer2 = nn.Dropout(self.dropout)
        self.head_mean = Head(batch_size = self.batch_size,
                              input_seq = self.mixer_input,
                              pred_seq = self.pred_seq,
                              channel = self.channel,
                              d_model = self.d_model)
        
        self.inp_dropout = nn.Dropout(self.input_dropout)
    
    # Series nMixers
    def forward(self, x):
        x = self.inp_dropout(x) # Batch, Length, feature
        x_emb = self.dropoutLayer(self.multi_embedding(x))
        self.x_emb = x_emb[:, -1:, :].clone()
        x_emb = self.norm0(x_emb) # Batch, Length, dmodel
        res = y = x_emb
        
        for i in range(self.num_mix):
            y = self.mixers[i](y)
        y = self.norm(res + self.dropoutLayer2(y))
        
        out = self.head_mean(y) # batch, pred_len, channel
        return out        
    

class CausalMixer(nn.Module):
    def __init__(self,
                 input_seq = [], 
                 out_seq = [], 
                 batch_size = [], 
                 channel = [], 
                 d_model = [], 
                 dropout = [], 
                 tfactor = [], 
                 dfactor = [], 
                 mixerType = []):
        
        super().__init__()
        self.input_seq = input_seq
        self.pred_seq = out_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.tfactor = tfactor # expansion factor for patch mixer
        self.dfactor = dfactor # expansion factor for embedding mixer
        
        self.tMixer = TemporalMixer(input_seq = self.input_seq, batch_size = self.batch_size, 
                                    channel = self.channel, pred_seq = self.pred_seq, 
                                    dropout = self.dropout, factor = self.tfactor,
                                    d_model = self.d_model, mixerType = mixerType)
        
        self.dropoutLayer = nn.Dropout(self.dropout)
        self.norm1 = nn.Sequential(Permute(0, -1, 1), nn.BatchNorm1d(self.d_model), Permute(0, -1, 1))
        self.norm2 = nn.Sequential(Permute(0, -1, 1), nn.BatchNorm1d(self.d_model), Permute(0, -1, 1))
        
        self.embeddingMixer = nn.Sequential(nn.Linear(self.d_model, int(self.d_model * self.dfactor)),
                                            nn.GELU(),
                                            nn.Dropout(self.dropout),
                                            nn.Linear(int(self.d_model * self.dfactor), self.d_model))
 
    def forward(self, x):
        res = x
        x = self.norm1(x + self.dropoutLayer(self.tMixer(x))) # batch, patch_num, d_model
        x = self.norm2(res + x + self.dropoutLayer(self.embeddingMixer(x))) # batch, patch_num, d_model
        return x 
    


class TemporalMixer(nn.Module):
    def __init__(self, input_seq = [], batch_size = [], channel = [], pred_seq = [],
                 dropout = [], factor = [], d_model = [], mixerType = []):
        
        super().__init__()
        self.input_seq = input_seq
        self.batch_size = batch_size
        self.channel = channel
        self.pred_seq = pred_seq
        self.dropout = dropout
        self.factor = factor
        self.d_model = d_model
        
        self.dropoutLayer = nn.Dropout(self.dropout)
        
        if mixerType == 'linear':
            self.layers = nn.Sequential(nn.Linear(self.input_seq, int(self.pred_seq * self.factor)),
                                        nn.GELU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(int(self.pred_seq * self.factor), self.pred_seq)
                                        )
        elif (mixerType == 'anti_causal') or (mixerType == 'causal'):
            self.layers = nn.Sequential(CausalLinearLayer(self.input_seq, int(self.pred_seq * self.factor), type_ = mixerType),
                                        nn.GELU(),
                                        nn.Dropout(self.dropout),
                                        CausalLinearLayer(int(self.pred_seq * self.factor), self.pred_seq, type_ = mixerType)
                                        )
        elif mixerType == 'none':
            self.layers = nn.Sequential() # None
        else:
            raise NotImplementedError('check mixer type')
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x



class Head(nn.Module):
    def __init__(self, batch_size = [], input_seq = [], pred_seq = [], channel = [], d_model = []):
        super().__init__()
        self.batch_size = batch_size
        self.input_seq = input_seq
        self.pred_seq = pred_seq
        self.channel = channel
        self.d_model = d_model
        self.head = nn.Linear(self.d_model, self.channel) # batch, dmodel, pred_length
                                  
    def forward(self, x):
        """
        The input and shape of x:  batch, length, dmodel
        output: batch, lenght, channel
        """
        out = self.head(x) # batch, length, channel
        return out # batch, channel, length

    
class MultiHeadEmbedding(nn.Module):
    def __init__(self, d_model = [], channel = [], nheads = [], clusterid = []):
        super().__init__()
        self.nheads = nheads
        self.d_model = d_model
        self.channel = channel
        cid = clusterid # cluster id for each channels: [cluster-x, cluster-y, cluster-z.....]
        assert self.nheads <= self.channel # Important
        
        self.ngroups = len(np.unique(cid)) # number of clusters
        self.groupidx = [] # list: [[channels in cluster-1], [channels in cluster-2],........]
        cid = np.array(cid)
        
        assert self.nheads == self.ngroups
        self.c = [] # list: [number_of_channles_in_cluster_1, number_of_channles_in_cluster_2, ...........]
        self.d = [] # list: [part_of_dmodel_alloted_to_cluster_1, part_of_dmodel_alloted_to_cluster_2, ...........]
        for i in range(self.ngroups):
            self.groupidx.append(list(np.where(cid == i)[0])) # list of the channels in cluster-i.
            self.c.append(len(self.groupidx[i]))
            self.d.append(int((len(self.groupidx[i]) / self.channel) * self.d_model)  ) # floor operation
        self.d[-1] += self.d_model - sum(self.d)
        print('Head intialization: clusters: {}'.format(self.c))
        self.layers = nn.ModuleList([nn.Linear(self.c[i], self.d[i]) for i in range(self.nheads)])        
        
        
    def forward(self, x):
        """
        x: [B, L, C]
        """
        B, L, C = x.shape
        y = torch.zeros((B, L, self.d_model), device = x.device, dtype = x.dtype)
        
        d_old = 0
        for i in range(self.nheads):
            d_start = d_old
            d_end = d_start + self.d[i]
            y[:, :, d_start:d_end] = self.layers[i](x[:, :, self.groupidx[i]])
            d_old = d_end
        return y

 