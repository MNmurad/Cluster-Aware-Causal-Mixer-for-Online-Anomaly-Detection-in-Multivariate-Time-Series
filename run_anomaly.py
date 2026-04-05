import os
import torch
import argparse

torch.set_printoptions(precision = 10)
from exp.exp_anomaly import Exp_Anomaly
import random 
from utils.Tuner import Tuner
from utils.tools import set_random_seed
from utils.anomaly_evaluation import getCombinedResult
from utils.config_loader import config_update
import time
from collections import defaultdict
import pandas as pd
from utils.unique_setting_name import unique_name
import sys, os, datetime
from utils.tools import printCombinedResult

print("=" * 40)
print("Time:       ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("Environment:", os.path.basename(os.path.dirname(sys.executable)))
print("Torch:      ", torch.__version__)
print("CUDA:       ", torch.version.cuda if torch.cuda.is_available() else "Not available")
print("=" * 40)

def main():
    parser = argparse.ArgumentParser(description = 'Time Series Anomaly detection')
    
    ''' frequent changing hy.params '''
    parser.add_argument('--model', type = str, required = False, default = 'Basic_Mixer', choices = ['Basic_Mixer', # proposed
                                                                                                     'Basic_Mixer_Temp', 'Model_diff', 
                                                                                                     'KAN_Mixer','SensitiveHUE', 'Uncertain_Mixer', 'WPMixer_SC', 
                                                                                                     'EDMixer_uncertain_SC', 'WPMixer', 'EDMixer', 'EDMixer_SC', 
                                                                                                     'GatedWPMixer', 'GatedEDMixer', 'TimesNet', 'xLSTMAD'], help = 'model of experiment')
    parser.add_argument('--data', type = str, default = 'WADI', choices = ['SWAT', 'PSM', 'WADI', 'SMAP','SMD', 'MSL', 
                                                                                    'SMAP_Combined', 'MSL_Combined', 'SMD_Combined'], help = 'dataset')
    parser.add_argument('--entity_id', type = int, required = False, default = -1, help = 'Entity ID number. 1 = first entity, n = nth entity, -1 = all entity')    
    
    
    
    parser.add_argument('--ablation', type = int, default = 0, help = 'ablation id 0 to n, 0 means no ablation')
    parser.add_argument('--mixerType', type = str, default = 'causal', choices = ['linear', 'causal', 'none'], help = 'causal, linear')
    

    parser.add_argument('--downsample', type = int, default = 1, help = 'median downsampling of the dataset')
    parser.add_argument('--target_optimization', type = str, default = 'sf1', choices = ['sf1', 'pf1'], help = 'sf1: sequence based anom detection, pf1: point based anom detection')
    parser.add_argument('--checkpoint_load', type = int, default = 1, help = '1: load, 0: no load')
    
    parser.add_argument('--num_workers', type = int, default = 0, help = 'data loader num workers')
    parser.add_argument('--task_name', type = str, required = False, choices = ['anomaly_detection'], default = 'anomaly_detection')
    parser.add_argument('--use_hyperParam_optim', action = 'store_true', default = False, help = 'True: HyperParameters optimization using optuna, False: no optimization')
    parser.add_argument('--train_flag', type = str, default = 'train', choices = ['train_full', 'train'], help = 'train_full: no validation, train: with validation')
    parser.add_argument('--seed', type = int, required = False, default = 42, help = 'random seed')    
    parser.add_argument('--detection_type', type = str, default = 'online', help = 'online detection and offline detection')
    parser.add_argument('--score_PAdjust', type = int, default = 0, help = '0: without point adjustment, 1: with point adjustment' )
    parser.add_argument('--featureSimilarity', type = int, default = 1, help = '0: KMeans, 1: Profile based Spec.Clust, ')
    
    # Fixed Params
    parser.add_argument('--batch_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('--train_epochs', type = int, default = 30, help = 'train epochs')
    
    # Tunable Params
    parser.add_argument('--seq_len', type = int, default = 24, help = 'length of the look back window')
    parser.add_argument('--pred_len', type = int, default = 1, help = 'prediction length')
    parser.add_argument('--num_mix', type = int, default = 1, help = 'number of mixer modules')
    parser.add_argument('--tfactor', type = int, default = 1, help = 'expansion factor in the patch mixer')
    parser.add_argument('--dfactor', type = int, default = 1, help = 'expansion factor in the embedding mixer')
    parser.add_argument('--learning_rate', type = float, default = 0.00498096466936417, help = 'initial learning rate')    
    parser.add_argument('--dropout', type = float, default = 0.05, help = 'dropout for mixer')
    parser.add_argument('--embedding_dropout', type = float, default = 0.05, help = 'dropout for embedding layer')
    parser.add_argument('--input_dropout', type = float, default = 0.0, help = 'dropout for input')
    parser.add_argument('--d_model', type = int, default = 40, help = 'embedding dimension')
    parser.add_argument('--weight_decay', type = float, default = 0.0, help = 'pytorch weight decay factor')
    parser.add_argument('--nheads', type = int, default = 5, help = 'number of heads in attention module') #################
    
    ''' Infrequent chaning parameters: Some of these has not used in our model '''
    parser.add_argument('--patch_flag', type = int, default = 0, help = '1: patch_enabled, 0: patch_disabled')
    parser.add_argument('--patch_len', type = int, default = 4, help = 'Patch size')
    parser.add_argument('--stride', type = int, default = 4, help = 'Stride')
    parser.add_argument('--lradj', type = str, default = 'type3', help = 'adjust learning rate')
    parser.add_argument('--use_multi_gpu', action = 'store_true', help = 'use multiple gpus', default = False)
    parser.add_argument('--n_jobs', type = int, required = False, choices = [1, 2, 3, 4], default = 1, help = 'number_of_jobs for optuna')
    parser.add_argument('--patience', type = int, default = 8, help = 'patience')
    parser.add_argument('--features', type = str, default = 'M', help = 'forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type = str, default = 'OT', help = 'target feature in S or MS task')
    parser.add_argument('--freq', type = str, default = 'h', help = 'freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type = str, default = './checkpoints/', help = 'location of model checkpoints')
    parser.add_argument('--cols', type = str, nargs = '+', default = None, help = 'certain cols from the data files as the input features')
    
    parser.add_argument('--itr', type = int, default = 1, help = 'experiments times')
    parser.add_argument('--use_amp', action = 'store_true', default = False, help = 'use automatic mixed precision training')
    parser.add_argument('--use_gpu', type = bool, default = True, help = 'use gpu')
    parser.add_argument('--gpu', type = int, default = 0, help = 'gpu')
    parser.add_argument('--devices', type = str, default = '0,1', help = 'device ids of multile gpus')
    parser.add_argument('--embed', type = str, default = 0)
    parser.add_argument('--loss', type = str, default = 'mse', choices = ['mse', 'smoothL1'])
    
    '''Parameters for TimesNet baseline'''
    # parser.add_argument('--seq_len', type = int, default = 24, help = 'length of the look back window')
    # parser.add_argument('--pred_len', type = int, default = 1, help = 'prediction length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')
    parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
    # parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    # parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    # parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    # parser.add_argument('--freq', type = str, default = 'h', help = 'freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    # parser.add_argument('--dropout', type = float, default = 0.05, help = 'dropout for mixer')
    
    # xLSTMAD
    parser.add_argument('--slstm_backend', type = str, default = 'vanilla', choices = ['vanilla', 'cuda'], help = 'the backend to use for the sLSTM layers, either "cuda" for GPU acceleration or "vanilla".')
    
    ''' Optuna Hyperparameters: if you don't pass the argument, then value form the hyperparameters_optuna.py will be considered as search region'''
    parser.add_argument('--optuna_seq_len', type = int, nargs = '+', required = False, default = None, help = 'Optuna seq length list')
    parser.add_argument('--optuna_lr', type = float, nargs = '+', required = False, default = None, help = 'Optuna lr: first-min, 2nd-max')
    parser.add_argument('--optuna_batch', type = int, nargs = '+', required = False, default = None, help = 'Optuna batch size list')
    parser.add_argument('--optuna_num_mix', type = int, nargs = '+', required = False, default = None, help = 'Optuna number of mixer')
    parser.add_argument('--optuna_tfactor', type = int, nargs = '+', required = False, default = None, help = 'Optuna tfactor list')
    parser.add_argument('--optuna_dfactor', type = int, nargs = '+', required = False, default = None, help = 'Optuna dfactor list')
    parser.add_argument('--optuna_epochs', type = int, nargs = '+', required = False, default = None, help = 'Optuna epochs list')
    parser.add_argument('--optuna_dropout', type = float, nargs = '+', required = False, default = None, help = 'Optuna dropout list')
    parser.add_argument('--optuna_input_dropout', type = float, nargs = '+', required = False, default = None, help = 'Optuna dropout list')
    parser.add_argument('--optuna_embedding_dropout', type = float, nargs = '+', required = False, default = None, help = 'Optuna embedding_dropout list')
    parser.add_argument('--optuna_dmodel', type = int, nargs = '+', required = False, default = None, help = 'Optuna dmodel list')
    parser.add_argument('--optuna_weight_decay', type = float, nargs = '+', required = False, default = None, help = 'Optuna weight_decay list')
    parser.add_argument('--optuna_trial_num', type = int, required = False, default = 2, help = 'Optuna trial number')        
    # Timesnet
    parser.add_argument('--optuna_top_k', type = int, nargs = '+', required = False, default = 3, help = 'Optuna top_k') 
    parser.add_argument('--optuna_d_ff', type = int, nargs = '+', required = False, default = 64, help = 'Optuna d_ff') 
    parser.add_argument('--optuna_e_layers', type = int, nargs = '+', required = False, default = 2, help = 'Optuna e_layers') 
    
    
    
    args = parser.parse_args()
    
    if args.model != 'Basic_Mixer':
        args.nheads = 1
        
    # Updating args for GPU 
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    data_parser = {
        'SWAT':  {'data': None, 'root_path': './data/SWAT/', 'T': 'FIT101', 'M': [51, 51], 'S': [None, None], 'MS': [51, 1], 'num_entities': 1},
        'PSM':  {'data': None, 'root_path': './data/PSM/', 'T': 'feature_24', 'M': [25, 25], 'S': [None, None], 'MS': [25, 1], 'num_entities': 1},
        'WADI':  {'data': 'WADI.pk', 'root_path': './data/WADI/', 'T': None, 'M': [123, 123], 'S': [None, None], 'MS': [None, None], 'num_entities': 1},
        'SMAP':  {'data': 'SMAP.pk', 'root_path': './data/SMAP_MSL', 'T': None, 'M': [25, 25], 'S': [None, None], 'MS': [None, None], 'num_entities': 55},
        'SMD':  {'data': 'SMD.pk', 'root_path': './data/SMD/', 'T': None, 'M': [38, 38], 'S': [None, None], 'MS': [None, None], 'num_entities': 28},
        'MSL':  {'data': 'MSL.pk', 'root_path': './data/SMAP_MSL', 'T': None, 'M': [55, 55], 'S': [None, None], 'MS': [None, None], 'num_entities': 27},
        'SMAP_Combined':  {'data': None, 'root_path': './data/SMAP_Combined', 'T': None, 'M': [25, 25], 'S': [None, None], 'MS': [None, None], 'num_entities': 1}, # entity 1 as combined
        'MSL_Combined':  {'data': None, 'root_path': './data/MSL_Combined', 'T': None, 'M': [55, 55], 'S': [None, None], 'MS': [None, None], 'num_entities': 1}, # entity 1 as combined
        'SMD_Combined':  {'data': None, 'root_path': './data/SMD_Combined/', 'T': None, 'M': [38, 38], 'S': [None, None], 'MS': [None, None], 'num_entities': 1}, # entity 1 as combined
    }
    
    # Updating args for data specific
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.root_path = data_info['root_path']
        args.target = data_info['T']
        args.c_in = data_info[args.features][0]
        args.enc_in = data_info[args.features][0] # Timesnet
        args.c_out = data_info[args.features][1]
        args.num_entities = data_info['num_entities']
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]
    
    # Condition: if we don't want hyperparam optimization
    if args.use_hyperParam_optim == False: 
        resultsList = []
        if args.entity_id == -1: # will consider all entities
        
            for args.entity_id in range(1, args.num_entities + 1): # updating the entity id. Now train and test will be run for this entity_id.
                args = config_update(args) # , data = args.data, entity_id = args.entity_id, downsample = args.downsample)
                results = trainTest(args)
                resultsList.append(results)

                
        elif args.entity_id > 0: # if we don't want to run the train test for all entities
            args = config_update(args) #, data = args.data, entity_id = args.entity_id, downsample = args.downsample)
            results = trainTest(args)
            resultsList.append(results)
            
        combined_results = getCombinedResult(resultsList)
        printCombinedResult(combined_results)
        save_result_to_excel(args, combined_results)
    
    # Condition: if we want hyperparam optimization
    elif args.use_hyperParam_optim:
        ''' Tuning the model using Optuna hyperparameter tuning framework'''
        assert args.entity_id > 0 # ensuring that entity id is not negative. negative entity id (-1) is used to run all entities.
        tuner = Tuner(42, args.n_jobs, args)
        tuner.tune(args, disable_pruner = False)
    return args


# total_params = 0
def trainTest(args):
    print('Args in experiment: {}'.format(args))
    setting = unique_name(**vars(args))
    
    set_random_seed(args.seed)
    Exp = Exp_Anomaly
    exp = Exp(args) 
    
    # params = exp.__parameter_count__()
    print('Start Training- {}'.format(setting))
    exp.train(setting, checkpoint_load = args.checkpoint_load)
    
    # ###########
    # global total_params
    # total_params += params
    # print('Number params: {}-{}'.format(params, total_params))
    # ################
    
    print('Start Testing- {}'.format(setting))
    test_results = exp.test(setting, full_anom_rslt = True)
    
    # Setting final results output
    test_results['anom_res']['gflops'] = exp.get_gflops()
    test_results['anom_res']['avgInfTime'] = test_results['avgInfTime']
    test_results['anom_res']['nParams'] = exp.__get_Trainable_nParams__()
    return test_results['anom_res']
    


def save_result_to_excel(args, result):
    while True:
        try:
            os.mkdir("save2.lock") 
            break
        except FileExistsError:
            time.sleep(2)
    try:
        __save_result__(args, result)
    finally:
        os.rmdir("save2.lock")


def __save_result__(args, result):
    if args.model == 'Basic_Mixer':
        columns = ['data', 'ablation', 'entity_id', 'downsample', 'nheads', 'pred_len', 'seq_len', 'learning_rate', 'batch_size',
                   'num_mix', 'mixerType', 'tfactor', 'dfactor', 'train_epochs', 'dropout', 'input_dropout',
                   'embedding_dropout', 'd_model', 'weight_decay', 'featureSimilarity', 'target_optimization', 'seed']
    elif args.model == 'TimesNet':
        columns = ['data', 'ablation', 'entity_id', 'downsample', 'pred_len', 'seq_len', 'learning_rate', 'batch_size',
                   'train_epochs', 'dropout', 'd_model', 'top_k', 'd_ff', 
                   'num_kernels', 'e_layers', 'weight_decay', 'target_optimization', 'seed']
    elif args.model == 'xLSTMAD':
        columns = ['data', 'ablation', 'entity_id', 'downsample', 'pred_len', 'seq_len', 'learning_rate', 'batch_size',
                   'train_epochs', 'dropout', 'd_model', 'weight_decay', 'target_optimization', 'seed']
        
    output_path = './outputs/Results/'
    excel_file_name = '{}'.format(args.data)
    excel_path = os.path.join(output_path, excel_file_name + '_res.xlsx')
    
    result_dic = defaultdict(list)
    for key, value in vars(args).items():
        if key in columns:
            result_dic[key].append(value)
            
    for key, value in result.items():
        result_dic[key].append(value)
    result_df = pd.DataFrame(result_dic)
    
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
        fallback_path = os.path.join(output_path, excel_file_name + '_' + timestamp + '.xlsx')
        result_df.to_excel(fallback_path, index=False)
        print(f"Saved to fallback file instead: {fallback_path}")
    return None


    
if __name__ == "__main__":
    main()