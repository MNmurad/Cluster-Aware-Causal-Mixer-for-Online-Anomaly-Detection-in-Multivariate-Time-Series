
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from utils.feature_clustering import features_clustering
from utils.downsampling import downsample_data

import warnings
warnings.filterwarnings('ignore')

class Dataset_PSM(Dataset):
    def __init__(self, root_path, flag = 'train', size = None, 
                  features = 'S', data_path = 'ETTh1.csv', 
                  target = 'OT', scale = True, inverse = False, 
                  timeenc = 0, freq = 'h', cols = None, seasonal_patterns = None, entity_id = 0,
                  cluster = None, downsample = 1, featureSimilarity = 1, **kwargs):

        self.seq_len = size[0]
        self.pred_len = size[2]
        self.entity_id = entity_id
        self.cluster = cluster
        self.featureSimilarity = featureSimilarity
        
        # init
        self.flag = flag
        assert self.flag in ['train', 'val', 'test', 'train_full'] # 'train_full' is to use whole dataset as train, no validation
        type_map = {'train':0, 'val':1, 'test':2, 'train_full': 0}
        self.set_type = type_map[self.flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        
        # we have separate test file
        self.vali_ratio = 0.2
        self.train_ratio = 1 if self.flag == 'train_full' else (1 - self.vali_ratio)
        self.test_ratio = 1 
        self.downsample = downsample
        
        self.norm_data_path = 'train.csv' # Normal dataset
        self.attk_data_path = 'test.csv' # Anomaly dataset
        self.label_path = 'test_label.csv'
        self.task = 'prediction_special' # 'reconstruction' , 'prediction', 'prediction_special'
        '''
        Reconstruction task and Prediction task: Reconstruction will reconstruct
        the input sequence. Prediction will predict the next seq of the input.
        '''
        self.__read_data__()

    def __read_data__(self):
        self.scaler = MinMaxScaler(feature_range = (-1, 1), copy = False) # StandardScaler()
        # self.scaler = StandardScaler()
        df_normal = pd.read_csv(os.path.join(self.root_path, self.norm_data_path))
        df_attack = pd.read_csv(os.path.join(self.root_path, self.attk_data_path))
        df_label = pd.read_csv(os.path.join(self.root_path, self.label_path))
        
        # df_normal = df_normal[0::10]
        # df_attack = df_attack[0::10]
        # df_label = df_label[0::10]
        
        # filling nan values
        df_normal = df_normal.fillna(method = 'ffill') # by observing, I found nan in only normal dataset
        
        # Normal/Attack label
        normal_label = np.zeros(df_normal.shape[0])
        attack_label = df_label['label'].values
        
        # column name correction: removing the space (It may not be necessary here)
        columns = {}
        for name in df_attack.columns:
            columns[name] = name.replace(' ', '')
        df_attack.rename(columns = columns, inplace = True)
        
        # column name correction: removing the space (It may not be necessary here)
        columns = {}
        for name in df_normal.columns:
            columns[name] = name.replace(' ', '')
        df_normal.rename(columns = columns, inplace = True)
        
        # cols = selected_columns # list(df_normal.columns); 
        cols = list(df_normal.columns); 
        cols.remove(self.target); 
        cols.remove('timestamp_(min)')
        
        # Reordering the data based on reordered columns
        df_normal = df_normal[['timestamp_(min)'] + cols + [self.target]]
        df_attack = df_attack[['timestamp_(min)'] + cols + [self.target]]

        num_train = int(len(df_normal) * self.train_ratio)
        
        # Multivariate/Univariate dataselection
        if self.features=='M' or self.features=='MS':
            _colsN_ = df_normal.columns[1:]
            _colsA_ = df_attack.columns[1:]
            df_normal_data = df_normal[_colsN_]
            df_attack_data = df_attack[_colsA_]
        elif self.features=='S':
            df_normal_data = df_normal[[self.target]]
            df_attack_data = df_attack[[self.target]]
        
        # Final train, test, validation data selection
        train_data = np.copy(df_normal_data[0: num_train].values.astype(np.float32))
        vali_data = np.copy(df_normal_data[num_train: -1].values.astype(np.float32))
        test_data = np.copy(df_attack_data.values.astype(np.float32))
        
        data_dic = {0: {'data': train_data, 'label': normal_label[0: num_train]},
                    1: {'data': vali_data, 'label': normal_label[num_train: -1]},
                    2: {'data': test_data, 'label': attack_label}}
        
        # Scaling
        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data_dic[self.set_type]['data'])
        else:
            data = data_dic[self.set_type]['data']
            
        f_dim = -1 if self.features=='MS' else 0
        
        # Pre Processing: Moving average
        data = self.__preProcessing__(data)
        
        # Downsampling
        data, label = downsample_data([data, data_dic[self.set_type]['label']], self.downsample)
        
        self.data_x = data # input data
        self.data_y = data[:,f_dim:] # target data
        
        self.data_x = torch.from_numpy(self.data_x).to(dtype = torch.float32)
        self.data_y = torch.from_numpy(self.data_y).to(dtype = torch.float32)
        self.data_label = torch.from_numpy(data_dic[self.set_type]['label']).to(dtype = torch.float32)
        
        ####### Additional for clustering ############
        if (self.flag == 'train_full') or (self.flag == 'train'):
            self.cluster_id = features_clustering(self.data_x, self.cluster, self.featureSimilarity)
            # self.cluster_id = features_clustering(self.data_x[50:-50, :], self.cluster, self.featureSimilarity)
        ####### End clustering #######################
        self.data_x = self.data_x.to('cuda')
        self.data_x = self.data_x.to('cuda')
        
        
    def __getitem__(self, index):
        stride_value = 1 
        seq_x, seq_y, seq_lab = self.__getitem_categorical__(index, stride = stride_value)
        return seq_x, seq_y, seq_lab
    
    def __len__(self):
        stride_value = 1
        return self.__len_categorical__(stride = stride_value)
    
    def __getitem_categorical__(self, index, stride = 1):
        '''
        For training, validation, window could be overlapping. so stride is 1.
        For testing, window should not be overlapping, so stride is set to pred_len.
        '''
        s_begin = index * stride
        s_end = s_begin + self.seq_len
        
        if self.task == 'prediction':
            r_begin = s_end
            r_end = r_begin + self.pred_len
        elif self.task == 'reconstruction':
            assert self.pred_len == self.seq_len
            r_begin = s_begin 
            r_end = r_begin + self.pred_len
        elif self.task == 'prediction_special':
            # r_begin = s_begin + self.seq_len // 2 # s_end - self.pred_len
            r_begin = s_end - self.pred_len
            r_end = r_begin + self.pred_len
            
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_lab = self.data_label[r_begin:r_end]
        
        return seq_x, seq_y, seq_lab

    
    def __len_categorical__(self, stride = 1):
        '''
        For training, validation, window could be overlapping. so stride is 1.
        For testing, window should not be overlapping, so stride is set to pred_len.
        '''
        if self.task == 'prediction':
            steps = ((len(self.data_x) - (self.seq_len + self.pred_len)) // stride) + 1 
            self.length = int((self.seq_len + self.pred_len) + (steps - 1) * stride)
        elif self.task == 'reconstruction':
            steps = ((len(self.data_x) - self.seq_len) // stride) + 1 
            self.length = int(self.seq_len + (steps - 1) * stride)
        elif self.task == 'prediction_special':
            steps = ((len(self.data_x) - (self.seq_len)) // stride) + 1 
            self.length = int((self.seq_len) + (steps - 1) * stride)
        return int(steps)

    def __preProcessing__(self, data):
        # x -> numpy array: length, feature
        # Moving average
        for i in range(data.shape[-1]):
            data[:, i] = np.convolve(data[:, i], np.ones(50)/50, mode = 'same')
            
        # clampping: basically as train is already betwn (-1, 1), so  this clamping will only effective to vali and test
        data[data > 4] = 4
        data[data < -4] = -4
        return data

    

    

    

