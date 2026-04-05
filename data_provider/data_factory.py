from data_provider.Dataset_PSM import Dataset_PSM  
from data_provider.Dataset_SWAT import Dataset_SWAT
from data_provider.Dataset_WADI import Dataset_WADI
from data_provider.Dataset_SMD import Dataset_SMD_Each_Entity
from data_provider.Dataset_SMAP import Dataset_SMAP_Each_Entity
from data_provider.Dataset_MSL import Dataset_MSL_Each_Entity
from data_provider.Dataset_SMAP_Combined import Dataset_SMAP_Combined
from data_provider.Dataset_MSL_Combined import Dataset_MSL_Combined
from data_provider.Dataset_SMD_Combined import Dataset_SMD_Combined
from torch.utils.data import DataLoader

data_dict = {
    'SWAT': Dataset_SWAT,
    'PSM': Dataset_PSM,
    'SMD': Dataset_SMD_Each_Entity,
    'SMAP': Dataset_SMAP_Each_Entity,
    'MSL': Dataset_MSL_Each_Entity,
    'WADI': Dataset_WADI,
    'SMAP_Combined': Dataset_SMAP_Combined,
    'MSL_Combined': Dataset_MSL_Combined,
    'SMD_Combined': Dataset_SMD_Combined
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    batch_size = args.batch_size
    freq = args.freq
    shuffle_flag = False if flag in ['test', 'val'] else True
    
    if args.task_name == 'anomaly_detection':
        drop_last = False
    elif args.task_name == 'long_term_forecast':
        drop_last = True

    data_set = Data(root_path=args.root_path,
                    data_path=args.data_path,
                    flag=flag,
                    size=[args.seq_len, None, args.pred_len], # [args.seq_len, args.label_len, args.pred_len]
                    features=args.features,
                    target=args.target,
                    timeenc=timeenc,
                    freq=freq,
                    entity_id = args.entity_id,
                    cluster = args.nheads,
                    downsample = args.downsample,
                    featureSimilarity = args.featureSimilarity,
                    )
    
    print(flag, len(data_set))
    
    if args.num_workers == 0:
        data_loader = DataLoader(data_set,
                                  batch_size=batch_size,
                                  shuffle=shuffle_flag,
                                  num_workers=args.num_workers,
                                  drop_last=drop_last)
    elif args.num_workers > 0:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory = True,
            persistent_workers = True)
    
    return data_set, data_loader




