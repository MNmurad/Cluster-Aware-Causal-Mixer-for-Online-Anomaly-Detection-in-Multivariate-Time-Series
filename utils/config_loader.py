import pandas as pd
import os
import argparse

def config_update(args):
    if args.model =='Basic_Mixer':
        args = __common_config_update__(args) # our model
    else:
        args = __common_config_update_otherbaselines__(args) # other baselines: TimesNet, xLSTMAD
        
    # if args.ablation == 0:
    #     args = __regular_config_update__(args)
    # elif args.ablation > 0:
    #     args = __config_update_for_ablation_study__(args)
    
    return args



def __common_config_update__(args):
    # For both regular and ablation studies
    config_path = os.path.join('./scripts/config/', 'config.xlsx')
    df = pd.read_excel(config_path)
    
    cols = ['data', 'entity_id', 'downsample', 'nheads', 
            'pred_len', 'seq_len', 'learning_rate', 'batch_size',
            'num_mix', 'tfactor', 'dfactor', 'train_epochs',
            'dropout', 'input_dropout', 'embedding_dropout', 'd_model', 
            'weight_decay', 'featureSimilarity', 'ablation', 'mixerType']
    
    df = df[cols]
    df = df[(df['data'] == args.data) & (df['entity_id'] == args.entity_id) & (df['ablation'] == args.ablation) & (df['mixerType'] == args.mixerType)]
    assert len(df) == 1 # checking whether df has only one row
    
    for key, value in df.items():
        if hasattr(args, key):
            val = value.iloc[0]  # or value.values[0]
            current_type = type(getattr(args, key))
            try:
                setattr(args, key, current_type(val))
            except Exception as e:
                print(f"Could not update '{key}' with value '{val}': {e}")

    print('*** Config Updated ***')
    return args


def __common_config_update_otherbaselines__(args):
    # For both regular and ablation studies
    config_path = os.path.join('./scripts/config/', f'config_{args.model}.xlsx')
    df = pd.read_excel(config_path)
    
    df = df[(df['data'] == args.data) & (df['entity_id'] == args.entity_id)]
    
    assert len(df) == 1 # checking whether df has only one row
    
    for key, value in df.items():
        if hasattr(args, key):
            val = value.iloc[0]  # or value.values[0]
            current_type = type(getattr(args, key))
            try:
                setattr(args, key, current_type(val))
            except Exception as e:
                print(f"Could not update '{key}' with value '{val}': {e}")

    print('*** Config Updated ***')
    return args


def __regular_config_update__(args):
    config_path = os.path.join('./scripts/config/', 'config.xlsx')
    df = pd.read_excel(config_path)
    
    cols = ['data', 'entity_id', 'downsample', 'nheads', 'pred_len',
           'seq_len', 'learning_rate', 'batch_size', 'num_mix', 'tfactor',
           'dfactor', 'train_epochs', 'dropout', 'input_dropout',
           'embedding_dropout', 'd_model', 'weight_decay', 'featureSimilarity']
    
    df = df[cols]
    
    df = df[(df['data'] == args.data) & (df['entity_id'] == args.entity_id)]
    assert len(df) > 0 # checking whether df has values
    
    for key, value in df.items():
        if hasattr(args, key):
            val = value.iloc[0]  # or value.values[0]
            current_type = type(getattr(args, key))
            try:
                setattr(args, key, current_type(val))
            except Exception as e:
                print(f"Could not update '{key}' with value '{val}': {e}")

    print('*** Config Updated ***')
    return args

    
def __config_update_for_ablation_study__(args):
    config_path = os.path.join('./scripts/config/', 'config.xlsx') # os.path.join('./outputs', 'config.xlsx')
    df = pd.read_excel(config_path)
    
    cols = ['data', 'entity_id', 'downsample', 'nheads', 'pred_len',
           'seq_len', 'learning_rate', 'batch_size', 'num_mix', 'tfactor',
           'dfactor', 'train_epochs', 'dropout', 'input_dropout',
           'embedding_dropout', 'd_model', 'weight_decay', 'featureSimilarity', 'ablation', 'mixerType']
    
    df = df[cols]
    
    df = df[(df['data'] == args.data) & (df['ablation'] == args.ablation)]
    
    assert len(df) > 0 # checking whether df has values
    
    for key, value in df.items():
        if hasattr(args, key):
            val = value.iloc[0]  # or value.values[0]
            current_type = type(getattr(args, key))
            try:
                setattr(args, key, current_type(val))
            except Exception as e:
                print(f"Could not update '{key}' with value '{val}': {e}")

    print('*** Ablation Config Updated ***')
    return args