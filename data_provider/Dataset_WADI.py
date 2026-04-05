
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from utils.feature_clustering import features_clustering
import pickle as pk
from utils.downsampling import downsample_data

class Dataset_WADI(Dataset):
    def __init__(self, root_path, flag = 'train', size = None, 
                 features = 'S', data_path = 'ETTh1.csv', 
                 target = 'OT', scale = True, inverse = False, 
                 timeenc = 0, freq = 'h', cols = None, entity_id = None,
                 cluster = None, downsample = 1, featureSimilarity = 1, **kwargs):

        self.seq_len = size[0]
        self.pred_len = size[2]
        self.entity_id = entity_id
        self.cluster = cluster
        self.root_path = root_path
        self.data_path = data_path
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
        self.cols = cols

        # we have separate test file
        self.vali_ratio = 0.2
        self.train_ratio = 1 if self.flag == 'train_full' else (1 - self.vali_ratio)
        self.test_ratio = 1 
        self.downsample = downsample
        
        self.task = 'prediction_special' # 'reconstruction' , 'prediction', 'prediction_special'
        self.__read_data__()


    def __read_data__(self):
        # self.scaler = StandardScaler()
        self.scaler = MinMaxScaler(feature_range = (-1, 1), copy = False)
        
        with open(os.path.join(self.root_path, self.data_path), "rb") as file:
            pk_data = pk.load(file)
        
        normal_data = pk_data['x_trn']
        normal_data = normal_data[21600:, :] # following the NPSR paper
        attack_data = pk_data['x_tst']
        attack_label = pk_data['lab_tst']
        normal_label = np.zeros(normal_data.shape[0])
        
        '''
        We just followed the same procedure followed by -
        Paper - "Nominality Score Conditioned Time Series Anomaly
        '''
        
        '''
        # Make 86-th column as 0. [following-"Nominality Score Conditioned Time Series Anomaly"]
        print('Make 86-th column as 0. This is critical.')
        However, they did not mention the column name of the 86 feature. 
        the column name is \\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2B_AIT_002_PV
        in my dataset, this column is 102.
        '''
        print('Make 102-th column as 0. This is critical.')
        normal_data[:, 102] = 0
        attack_data[:, 102] = 0
        
        num_train = int(normal_data.shape[0] * self.train_ratio)
        train_data = np.copy(normal_data[0: num_train, :].astype(np.float32))
        vali_data = np.copy(normal_data[num_train: -1, :].astype(np.float32))
        test_data = np.copy(attack_data.astype(np.float32))

        data_dic = {0: {'data': train_data, 'label': normal_label[0: num_train], 'time': None},
                    1: {'data': vali_data, 'label': normal_label[num_train: -1], 'time': None},
                    2: {'data': test_data, 'label': attack_label, 'time': None}}
        
        # Scaling
        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data_dic[self.set_type]['data'])
        else:
            data = data_dic[self.set_type]['data']
            
        f_dim = -1 if self.features=='MS' else 0
        
        # Pre Processing: 
        data = self.__preProcessing__(data) # only clamping
        
        # Downsampling
        data, label = downsample_data([data, data_dic[self.set_type]['label']], self.downsample)
        self.data_x = data
        self.data_y = data[:,f_dim:] # target
        
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
        # x -> numpy array: length, feature
        # # # Moving average
        # for i in range(data.shape[-1]):
        #     data[:, i] = np.convolve(data[:, i], np.ones(50)/50, mode = 'same')
            
        # clampping: basically as train is already betwn (-1, 1), so  this clamping will only effective to vali and test
        data[data > 4] = 4
        data[data < -4] = -4
        return data
    