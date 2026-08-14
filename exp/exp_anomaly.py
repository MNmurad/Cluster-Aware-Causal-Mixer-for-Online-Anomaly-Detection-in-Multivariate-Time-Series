# -*- coding: utf-8 -*-
"""
Created when I should've been asleep
@author: Murad
SISLab, USF
mmurad@usf.edu
"""

from models.CCM_TAD import CCM_TAD
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.anomaly_evaluation import anom_evaluation
from data_provider.data_factory import data_provider
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import optuna
from thop import profile
import pickle as pk
import warnings
warnings.filterwarnings('ignore')


class Exp_Anomaly():
    def __init__(self, args):
        self.args = args
        self.optuna_optim_target = np.inf
        self.min_loss = np.inf
        self.best_epoch = -1
        assert self.args.task_name == 'anomaly_detection'
        print('task: anomaly_detection')
        
        self.device = self._acquire_device()
        self.args.device = self.device
        
        temp_data, _ = self._get_data(flag = 'train_full') # called this to just get the cluster id
        self.args.clusterid = temp_data.cluster_id
        self.model = self._build_model().to(self.device)
        
        
    def _build_model(self):
        model_dict = {'CCM_TAD': CCM_TAD}
        model = model_dict[self.args.model](self.args).float()
            
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    
    def __get_Trainable_nParams__(self):
        pytorch_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return pytorch_trainable_params
    
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    
    def _get_data(self, flag = None):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), 
                                 lr = self.args.learning_rate, 
                                 weight_decay = self.args.weight_decay)
        return model_optim
    
    
    def _select_criterion(self):
        assert self.args.loss == 'mse'
        criterion = nn.MSELoss()
        return criterion
    
    
    def __parameter_count__(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return trainable
        
    
    def train(self, setting, optunaTrialReport = None, checkpoint_load = False):
        train_data, train_loader = self._get_data(flag = self.args.train_flag)
        test_data, test_loader = self._get_data(flag = 'test')
        vali_data, vali_loader = self._get_data(flag = 'val') if (self.args.train_flag == 'train') else (None, None)
        
        os.makedirs(path := os.path.join(self.args.checkpoints, setting), exist_ok = True)
        if checkpoint_load:
            self.__load_state_dict__(path)
            print('>>> checkpoint_loading_Successful <<<')
            return self.model
   
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion() 

        if self.args.use_amp:
            scaler =  torch.cuda.amp.GradScaler(init_scale = 1024)
        
        for epoch in range(self.args.train_epochs):
            train_loss = []
            epoch_time = time.time()
            self.model.train()
            
            for i, (batch_x, batch_y, label) in enumerate(train_loader):
                if self.args.use_amp:
                    pred_mean, true, _, _ = self._process_one_batch(train_data, batch_x, batch_y, label, 'train')
                    with torch.cuda.amp.autocast():
                        loss = criterion(pred_mean, true)
                    model_optim.zero_grad(set_to_none = True)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    pred_mean, true, _, _ = self._process_one_batch(train_data, batch_x, batch_y, label, 'train')
                    loss = criterion(pred_mean, true) 
                    model_optim.zero_grad(set_to_none = True)
                    loss.backward()
                    model_optim.step()
                
                train_loss.append(loss.detach())
            
            train_loss = torch.tensor(train_loss).mean().item()
            vali_mse, vali_outputs = self.vali(setting, data = vali_data, loader = vali_loader)
            test_results = self.test(setting, test_data = test_data, test_loader = test_loader, 
                                     vali_outputs = vali_outputs, save_output = False, full_anom_rslt = False)
            
            # ########################### this part is just for optuna ###########
            assert self.args.target_optimization in ['sf1', 'pf1']
            current_value_optuna_target = -test_results['f1'] # result depending on target optimization
            
            if (current_value_optuna_target <  self.optuna_optim_target) and epoch >= 1: # will not consider any value from epoch 1
                self.optuna_optim_target = current_value_optuna_target
                self.best_epoch = epoch
                self.best_result = test_results['anom_res'] # saving the result for tuner data saving
                self.__save_state_dict__(path)
                
            if optunaTrialReport is not None:
                optunaTrialReport.report(self.optuna_optim_target, epoch)
                if optunaTrialReport.should_prune():
                    raise optuna.exceptions.TrialPruned()
            #############################################################
            
            print("Epoch {}: cost time: {:.2f} sec".format(epoch, time.time()-epoch_time))
            marking = '*' if self.best_epoch == epoch else ''
            print(f"\tTr: {train_loss:.3f} Val: {vali_mse:.3f} Tst: {test_results['mse']:.3f} Tst.{self.args.target_optimization}: {test_results['f1']:.3f} a: {test_results['anom_res']['alpha']} {marking}")
            adjust_learning_rate(model_optim, None, epoch + 1, self.args)
            
        self.__load_state_dict__(path) # loadding best model at the end
        print('Best Epoch: {}, Best F1: {}'.format(self.best_epoch, -self.optuna_optim_target))
        return self.model
    

    def test(self, setting, test_data = None, test_loader = None, vali_outputs = None, save_output = True, full_anom_rslt = True):
        self.model.eval()
        if test_data is None:
            test_data, test_loader = self._get_data(flag = 'test')
            vali_data, vali_loader = self._get_data(flag = 'val')
            _, vali_outputs = self.vali(setting, data = vali_data, loader = vali_loader)
            
        test_preds, test_trues, test_labels, inf_time = [], [], [], 0
        with torch.no_grad():
            for i, (batch_x, batch_y, label) in enumerate(test_loader):
                
                t1 = time.time()
                pred, true, test_label, _ = self._process_one_batch(test_data, batch_x, batch_y, label, 'test')
                inf_time += (time.time() - t1)
                
                test_preds.append(pred.cpu())
                test_trues.append(true.cpu())
                test_labels.append(test_label)

            test_preds = torch.cat(test_preds)
            test_trues = torch.cat(test_trues) 
            test_labels = np.concatenate(test_labels, axis = 0)
            inf_time = inf_time / (i + 1) # Seconds: average inference time per batch

            test_preds = test_preds.reshape(-1, test_preds.shape[-1])
            test_trues = test_trues.reshape(-1, test_trues.shape[-1])
            test_labels = test_labels.reshape(-1)
            mae, mse, rmse, mape, mspe = metric(test_preds.numpy(), test_trues.numpy())
            
            result = anom_evaluation(self.args, test_preds, test_trues, test_labels, 
                                     self.args.target_optimization, vali_outputs = vali_outputs, 
                                     full_anom_rslt = full_anom_rslt)
            
            if save_output:
                os.makedirs(data_outputs :='./outputs/data_outputs/', exist_ok = True)
                with open(os.path.join(data_outputs, f'outputs_{self.args.data}_{self.args.entity_id}_{self.args.target_optimization}_ab{self.args.ablation}.pk'), 'wb') as file:
                    pk.dump({'preds': test_preds, 'trues': test_trues, 'labels': test_labels, 'result': result, 
                              'clusterid': self.args.clusterid}, file)
                
        return {'mse': mse, 'mae': mae, 'f1': result['F1'], 'avgInfTime': inf_time, 'anom_res': result}
    
    
    def vali(self, setting, data = None, loader = None):
        if data is None:
            return np.inf, {'nominal_score': [], 'preds': [], 'trues': []}
        self.model.eval()
        
        preds, trues, = [], []
        with torch.no_grad():
            for i, (batch_x, batch_y, label) in enumerate(loader):
                pred, true, _, _ = self._process_one_batch(data, batch_x, batch_y, label, 'vali')
                preds.append(pred.cpu())
                trues.append(true.cpu())
    
            preds = torch.cat(preds)
            trues = torch.cat(trues) 
            mse = ((preds - trues) ** 2).mean()
            score_list = ((preds - trues) ** 2).mean(-1)
            nominal_score = score_list.squeeze().sort()[0]
            nominal_score = nominal_score.detach().cpu().numpy()
            vali_output = {'nominal_score': nominal_score, 'preds': preds.numpy(), 'trues': trues.numpy()}
        return mse, vali_output


    def __load_state_dict__(self, path):
        load_checkpoint = path + '/' + 'checkpoint.pth'
        loaded_state_dicts = torch.load(load_checkpoint, map_location = self.device) # torch.load(load_checkpoint)
        self.model.load_state_dict(loaded_state_dicts)
        return 
    
    
    def __save_state_dict__(self, path):
        state_dicts = self.model.state_dict()
        torch.save(state_dicts, path + '/' + 'checkpoint.pth')
        return
    
    
    def get_gflops(self):
        batch = self.args.batch_size
        seq = self.args.seq_len
        channel = self.args.c_in
        input_tensor = torch.randn(batch, seq, channel).to(device = self.device) # Dumy inputs
        
        self.model.eval()
        macs, params = profile(self.model, inputs = (input_tensor, ), verbose = True)
        gflops = 2 * macs / 1e9  # convert to GFLOPs
        print(f"Total GFLOPs: {gflops:.4f}")
        return gflops
    

    def _process_one_batch(self, dataset_object, batch_x, target, label, function):
        batch_x = batch_x.to(device = self.device)
        target =  target.to(device = self.device)
        
        pred = self.model(batch_x)
        return pred, target, label, batch_x
    
    