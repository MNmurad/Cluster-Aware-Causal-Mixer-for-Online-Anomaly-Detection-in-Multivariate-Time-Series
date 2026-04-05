
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pickle as pk
from utils.downsampling import downsample_data

from utils.feature_clustering import features_clustering
import warnings
warnings.filterwarnings('ignore')


class Dataset_SMAP_Each_Entity(Dataset):
    def __init__(self,  root_path = [], data_path = [], flag = 'train', 
                 size = None, scale = True, entity_id = None, 
                 cluster = None, downsample = 1, featureSimilarity = 1, **kwargs):
        self.seq_len = size[0]
        self.pred_len = size[2]
        self.entity_id = entity_id # starts from 1
        self.cluster = cluster
        self.root_path = root_path
        self.data_path = data_path
        self.featureSimilarity = featureSimilarity
        
        self.flag = flag
        assert self.flag in ['train', 'val', 'test', 'train_full'] # 'train_full' is to use whole dataset as train, no validation
        type_map = {'train':0, 'val':1, 'test':2, 'train_full': 0}
        self.set_type = type_map[self.flag]
        self.scale = scale
        
        self.vali_ratio = 0.2
        self.train_ratio = 1 if self.flag == 'train_full' else (1 - self.vali_ratio)
        self.test_ratio = 1 
        self.downsample = downsample
        
        self.task = 'prediction_special' # 'reconstruction' , 'prediction', 'prediction_special'
        self.__read_data__()

    def __read_data__(self):
        self.scaler = MinMaxScaler(feature_range = (-1, 1), copy = False) # StandardScaler()
        
        pk_path = os.path.join(self.root_path, self.data_path)
        with open(pk_path, 'rb') as file:
            data = pk.load(file)

        data_normal = data['x_trn'][self.entity_id - 1].astype(np.float32) # entity id starts from 1, however index strt from 0
        data_attack = data['x_tst'][self.entity_id - 1].astype(np.float32) # entity id starts from 1, however index strt from 0
        attack_label = data['lab_tst'][self.entity_id - 1].astype(np.int32) # entity id starts from 1, however index strt from 0
        normal_label = np.zeros(data_normal.shape[0]).astype(np.int32)

        num_train = int(len(data_normal) * self.train_ratio)
        
        # Final train, test, validation data selection
        train_data = np.copy(data_normal[0: num_train, :]) ################## Important: copy##########
        vali_data = np.copy(data_normal[num_train: -1, :]) ################## Important: copy##########
        test_data = np.copy(data_attack) ################## Important: copy##########
        
        data_dic = {0: {'data': train_data, 'label': normal_label[0: num_train]},
                    1: {'data': vali_data, 'label': normal_label[num_train: -1]},
                    2: {'data': test_data, 'label': attack_label}}
        
        # Scaling
        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data_dic[self.set_type]['data'])
        else:
            data = data_dic[self.set_type]['data']
        
        # Pre Processing: Moving average
        data = self.__preProcessing__(data)
        
        # Downsampling
        data, label = downsample_data([data, data_dic[self.set_type]['label']], self.downsample)
        
        self.data_x = data # input data
        self.data_y = data # target data
        
        self.data_x = torch.from_numpy(self.data_x).to(dtype = torch.float32)
        self.data_y = torch.from_numpy(self.data_y).to(dtype = torch.float32)
        self.data_label = torch.from_numpy(label).to(dtype = torch.float32)
        
        ####### Additional for clustering ############
        if (self.flag == 'train_full') or (self.flag == 'train'):
            self.cluster_id = features_clustering(self.data_x, self.cluster, self.featureSimilarity)
            # self.cluster_id = features_clustering(self.data_x[50:-50, :], self.cluster, self.featureSimilarity)
        ####### End clustering #######################
        
        
    def __getitem__(self, index):
        stride_value = 1 # self.pred_len if self.flag == 'test' else 1
        seq_x, seq_y, seq_lab = self.__getitem_categorical__(index, stride = stride_value)
        return seq_x, seq_y, seq_lab
    
    
    def __len__(self):
        stride_value = 1 # self.pred_len if self.flag == 'test' else 1
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
        # # x -> numpy array: length, feature
        # # Moving average
        # for i in range(data.shape[-1]):
        #     data[:, i] = np.convolve(data[:, i], np.ones(50)/50, mode = 'same')
            
        # clampping: basically as train is already betwn (-1, 1), so  this clamping will only effective to vali and test
        data[data > 4] = 4
        data[data < -4] = -4
        return data
    
    
def Dataset_SMAP_Combined(root_path = None, data_path = None, flag = None, 
             size = None, scale = None, entity_id = None, 
             cluster = None, downsample = 1, featureSimilarity = 1,  **kwargs):
    nentities = 55
    dataset_list = []
    for i in range(nentities):
        dataset_single_entity = Dataset_SMAP_Each_Entity(root_path = root_path, data_path = data_path, flag = flag, 
                     size = size, scale = scale, entity_id = entity_id, 
                     cluster = cluster, downsample = 1, featureSimilarity = 1)
        dataset_list.append(dataset_single_entity)
    return ConcatDataset(dataset_list)