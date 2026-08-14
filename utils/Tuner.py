# -*- coding: utf-8 -*-
"""
Created when I should've been asleep
@author: Murad
SISLab, USF
mmurad@usf.edu
"""

import torch
from utils.tools import set_random_seed
import optuna
from collections import defaultdict
import pandas as pd
torch.set_printoptions(precision = 10)
from exp.exp_anomaly import Exp_Anomaly
import datetime
import os
import uuid
import time
import numpy as np
from utils.unique_setting_name import unique_name
from optuna.visualization import plot_param_importances, plot_optimization_history, plot_parallel_coordinate, plot_contour, plot_slice
import warnings


class Tuner:
    # Tuner for Long Term forecasting
    def __init__(self, ranSeed, n_jobs, args):
        self.fixedSeed = ranSeed
        self.n_jobs = n_jobs 
        self.min_optuna_optim_target = np.inf

        self.result_dic = defaultdict(list)
        self.current_time = datetime.datetime.now()
        self.current_time = str(self.current_time).replace(':', '-').split('.')[0]
        self.output_path = './outputs/Anomaly_HyperParam_Search/'
        os.makedirs(output_path2 := os.path.join(self.output_path, f"{args.model}_{args.data}_{self.current_time}"), exist_ok = True)
        self.output_path2 = output_path2


    def optuna_objective(self, trial, args):
        # # these are the params that will be tuned:
        args.seq_len = trial.suggest_categorical('seq_len', args.optuna_seq_len)
        args.learning_rate = trial.suggest_loguniform('learning_rate', args.optuna_lr[0], args.optuna_lr[1]) 
        args.batch_size = trial.suggest_categorical('batch_size', args.optuna_batch)
        args.train_epochs = trial.suggest_categorical('train_epochs', args.optuna_epochs) 
        args.d_model = trial.suggest_categorical('d_model', args.optuna_dmodel) 
        args.dropout = trial.suggest_categorical('dropout', args.optuna_dropout) 
        args.weight_decay = trial.suggest_categorical('weight_decay', args.optuna_weight_decay) 

        args.num_mix = trial.suggest_categorical('num_mix', args.optuna_num_mix) 
        args.tfactor = trial.suggest_categorical('tfactor', args.optuna_tfactor) 
        args.dfactor = trial.suggest_categorical('dfactor', args.optuna_dfactor) 
        args.input_dropout = trial.suggest_categorical('input_dropout', args.optuna_input_dropout) 
        args.embedding_dropout = trial.suggest_categorical('embedding_dropout', args.optuna_embedding_dropout) 

        setting = unique_name(**vars(args))
        set_random_seed(self.fixedSeed) # 42
        
        Exp = Exp_Anomaly
        exp = Exp(args) # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        print('Args in experiment: {}'.format(args))
        exp.train(setting, optunaTrialReport = trial)        
        
        ########################################################################################
        if exp.optuna_optim_target < self.min_optuna_optim_target:
            self.min_optuna_optim_target = exp.optuna_optim_target # F1 value
            self.path = f"checkpoints/Optuna/{setting}_{int(time.time())}_trial_{trial.number}/"
            os.makedirs(self.path, exist_ok = True)
            torch.save(exp.model.state_dict(), os.path.join(self.path, 'checkpoint.pth'))
        ########################################################################################
        return exp.optuna_optim_target
    
    
    def tune(self, args, disable_pruner = True): # args with some fixed params. other will be tuned in objective function
        n = args.optuna_trial_num

        try:
            del self.study
            print('deleted previous tuner obj')
        except:
            print('no prev tuner obj')
        pruner = optuna.pruners.NopPruner() if disable_pruner else optuna.pruners.MedianPruner()
        self.study = optuna.create_study(direction='minimize', sampler = optuna.samplers.TPESampler(seed = 42), pruner = pruner)
        wrapped_objective = lambda trial: self.optuna_objective(trial, args)
        
        self.study.optimize(wrapped_objective, n_trials = n, n_jobs = self.n_jobs, callbacks = [self.save_optuna_stat]) 
        
        #####################################
        while True:
            try:
                os.mkdir("save.lock") 
                break
            except FileExistsError:
                time.sleep(2)
        try:
            self.save_result(args)#, result)
        finally:
            os.rmdir("save.lock")
        #####################################
        return 
    
    
    def save_optuna_stat(self, study, trial):
        if trial.number == 0: # some stat can not evaluate in the first trial
            return
        hyp_path = self.output_path2
        
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message = ".*unique value length is less than 2.*", category = UserWarning)
            try:
                #
                fig1 = plot_param_importances(study)
                fig1.write_html(os.path.join(hyp_path, "param_importances.html"))
                #
                fig2 = plot_optimization_history(study)
                fig2.write_html(os.path.join(hyp_path, "optimization_history.html"))
                #
                fig3 = plot_parallel_coordinate(study)
                fig3.write_html(os.path.join(hyp_path, "parallel_coordinate.html"))
                #
                fig4 = plot_contour(study)
                fig4.write_html(os.path.join(hyp_path, "contour.html"))
                #
                fig5 = plot_slice(study)
                fig5.write_html(os.path.join(hyp_path, "slice.html"))
            except:
                print('optuna_stat_error')
        optuna.logging.set_verbosity(optuna.logging.INFO)
        return
    
    def save_result(self, args):#, result):
        # Loading best parameters ##########
        best_params = self.study.best_params
        for key, value in best_params.items():
            setattr(args, key, value)
        # Loading finish ###################
        
        # Running the test function with optimum params
        setting = unique_name(**vars(args))
        set_random_seed(self.fixedSeed) # 42
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        exp = Exp_Anomaly(args) 
        exp.__load_state_dict__(self.path)
        out = exp.test(setting) 
        results = out['anom_res'] # anomaly detection results
        
        data, pred_len = args.data, args.pred_len
        best_result = self.study.best_trial.value
       
        # ################# Tuner Final output Results ###########################
        self.result_dic['data'].append(data)
        self.result_dic['entity_id'].append(args.entity_id)
        self.result_dic['ablation'].append(args.ablation)
        self.result_dic['downsample'].append(args.downsample)
        self.result_dic['target_value'].append(best_result)
        self.result_dic['nheads'].append(args.nheads)
        self.result_dic['pred_len'].append(pred_len)
        self.result_dic['mixerType'].append(args.mixerType)
        for key, value in best_params.items():
            self.result_dic[key].append(value)
        self.result_dic['featureSimilarity'].append(args.featureSimilarity)
        self.result_dic['TP'].append(results['TP'])
        self.result_dic['FP'].append(results['FP'])
        self.result_dic['TN'].append(results['TN'])
        self.result_dic['FN'].append(results['FN'])
        self.result_dic['PR_AUC'].append(results['PR_AUC'])
        self.result_dic['alphas'].append(results['alpha'])
        result_df = pd.DataFrame(self.result_dic)
        print(result_df)
        
        ##### Tuner output saving to specific folder ###################
        try:
            result_df.to_excel(os.path.join(self.output_path2, 'hyperparameters.xlsx'))
        except:
            print('save failed: close best param csv file')

        os.makedirs(checkpoint_path := os.path.join(self.output_path2, setting), exist_ok = True)
        exp.__save_state_dict__(checkpoint_path) # saving the best checkpoint to the specific hypertuning folder for the model and data
        #################### END  ######################################
        
        ##### Tuner output saving to a Common Data File ###################
        excel_file_name = '{}'.format(args.data)
        excel_path = os.path.join(self.output_path, excel_file_name + '.xlsx')
        try:
            if os.path.exists(excel_path):
                existing_df = pd.read_excel(excel_path)
                updated_df = pd.concat([existing_df, result_df], ignore_index = True)
            else:
                updated_df = result_df
            updated_df.to_excel(excel_path, index = False) # , float_format="%.17e")
        
        except Exception as e:
            print(f"Could not write to {excel_path} due to: {e}")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            fallback_path = os.path.join(self.output_path, excel_file_name + '_' + timestamp + '.xlsx')
            result_df.to_excel(fallback_path, index=False)
            print(f"Saved to fallback file instead: {fallback_path}")
        print(updated_df)
        


def generate_unique_filename(prefix="optuna_log", extension=".txt"):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]  # Short UUID
    return f"{prefix}_{timestamp}_{unique_id}{extension}"





            