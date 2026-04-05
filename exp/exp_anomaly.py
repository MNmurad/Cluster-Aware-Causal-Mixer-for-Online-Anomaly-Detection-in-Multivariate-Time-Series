from models.basic_mixer import Basic_Mixer
try:
    from models.xlstmad import xLSTMAD
except:
    xLSTMAD = None

try:
    from models.TimesNet import Model as TimesNet
except:
    TimesNet = None
    
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
        model_dict = {'Basic_Mixer': Basic_Mixer,
                      'TimesNet': TimesNet,
                      'xLSTMAD': xLSTMAD}
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
        model_optim = optim.Adam(self.model.parameters(), lr = self.args.learning_rate, weight_decay = self.args.weight_decay)
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
        
        # train_loss_epochs = []; vali_loss_epochs = []
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
            test_results = self.test(setting, test_data = test_data, test_loader = test_loader, vali_outputs = vali_outputs, save_output = False, full_anom_rslt = False)
            
            # train_loss_epochs.append(train_loss)
            # vali_loss_epochs.append(vali_mse)
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
    

    # Test Normal
    def test(self, setting, test_data = None, test_loader = None, vali_outputs = None, save_output = True, full_anom_rslt = True):
        self.model.eval()
        if test_data is None:
            test_data, test_loader = self._get_data(flag = 'test')
            vali_data, vali_loader = self._get_data(flag = 'val')
            _, vali_outputs = self.vali(setting, data = vali_data, loader = vali_loader)
            # np.save(f'vali_score_{self.args.data}_{self.args.entity_id}.npy', vali_outputs['nominal_score'])
            # vali_outputs = np.load(f'./outputs/data_outputs/{setting}_val_score.npz')
        
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
            
            
            result = anom_evaluation(self.args, test_preds, test_trues, test_labels, self.args.target_optimization, vali_outputs = vali_outputs, full_anom_rslt = full_anom_rslt)
            
            # if self.args.target_optimization == 'sf1':
            #     os.makedirs('./custom/', exist_ok=True)
            #     np.savez(f'./custom/vali_score_{self.args.data}_{self.args.entity_id}.npz', nominal_score = vali_outputs['nominal_score'], alpha = result['alpha'])
            
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
    
    
# def corr_difference_torch(preds, trues):
#     if len(preds.shape) == 2:
#         # Compute correlation matrices using PyTorch
#         pred_corr = torch.corrcoef(preds.T)
#         true_corr = torch.corrcoef(trues.T)
        
#         # Replace NaN with 0 for constant features
#         true_corr = torch.nan_to_num(true_corr, nan=0.0)
        
#         # Compute mean absolute difference
#         avg_abs_error_corr = (pred_corr - true_corr).abs().mean(dim=1)
#     else:
#         # preds = preds.permute(0, 2, 1)
#         # trues = trues.permute(0, 2, 1)
#         # avg_abs_error_corr = []
#         # for i in range(preds.shape[0]):
#         #     pred_corr = torch.corrcoef(preds[i, :, :])
#         #     true_corr = torch.corrcoef(trues[i, :, :])
#         #     true_corr = torch.nan_to_num(true_corr, nan=0.0)
#         #     avg_abs_error_corr.append((pred_corr - true_corr).abs().mean(dim=1))
#         # avg_abs_error_corr = torch.stack(avg_abs_error_corr, dim = -1).mean(-1)

#         # preds, trues: [B, T, F]
#         preds = preds.permute(0, 2, 1)  # [B, F, T]
#         trues = trues.permute(0, 2, 1)  # [B, F, T]
    
#         # Center each feature
#         preds_c = preds - preds.mean(dim=-1, keepdim=True)
#         trues_c = trues - trues.mean(dim=-1, keepdim=True)
    
#         # Compute covariance: [B, F, F]
#         cov_pred = preds_c @ preds_c.transpose(-1, -2) / (preds_c.shape[-1] - 1)
#         cov_true = trues_c @ trues_c.transpose(-1, -2) / (trues_c.shape[-1] - 1)
    
#         # Compute std: [B, F]
#         std_pred = preds_c.pow(2).mean(dim=-1).sqrt()
#         std_true = trues_c.pow(2).mean(dim=-1).sqrt()
    
#         # std outer product → [B, F, F]
#         denom_pred = std_pred.unsqueeze(-1) * std_pred.unsqueeze(-2)
#         denom_true = std_true.unsqueeze(-1) * std_true.unsqueeze(-2)
    
#         # Avoid divide-by-zero
#         denom_pred = torch.nan_to_num(denom_pred, nan=1e-6)
#         denom_true = torch.nan_to_num(denom_true, nan=1e-6)
    
#         # Correlation matrices
#         corr_pred = cov_pred / denom_pred
#         corr_true = cov_true / denom_true
    
#         # Replace NaN
#         corr_true = torch.nan_to_num(corr_true, nan=0.0)
    
#         # Absolute difference, averaged over F x F
#         avg_abs_error_corr = (corr_pred - corr_true).abs().mean(dim=(1, 2))  # [B]
    
#         # return avg_abs_error_corr.mean()  # overall average

#     return avg_abs_error_corr

