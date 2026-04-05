
def unique_name(**kwargs):
    if kwargs['model'] == 'Basic_Mixer':
        setting = '{}_{}_{}_ds{}_E{}_H{}_{}-{}_d{}_b{}_f{}-{}_{}{}_fs{}_{}'.format(
            kwargs['model'],
            kwargs['data'],
            kwargs['ablation'],
            kwargs['downsample'],
            kwargs['entity_id'],
            kwargs['nheads'],
            kwargs['seq_len'],
            kwargs['pred_len'],
            kwargs['d_model'],
            kwargs['batch_size'],
            kwargs['tfactor'],
            kwargs['dfactor'],
            kwargs['mixerType'],
            kwargs['num_mix'],
            kwargs['featureSimilarity'],
            kwargs['seed'],
            )
    elif kwargs['model'] == 'TimesNet':
        setting = '{}_{}_{}_ds{}_E{}_{}-{}_d{}_b{}_tk{}_df{}_nk{}_el{}_{}'.format(
            kwargs['model'],
            kwargs['data'],
            kwargs['ablation'],
            kwargs['downsample'],
            kwargs['entity_id'],
            kwargs['seq_len'],
            kwargs['pred_len'],
            kwargs['d_model'],
            kwargs['batch_size'],
            kwargs['top_k'],
            kwargs['d_ff'],
            kwargs['num_kernels'],
            kwargs['e_layers'],
            kwargs['seed'],
            )
    elif kwargs['model'] == 'xLSTMAD':
        setting = '{}_{}_{}_ds{}_E{}_{}-{}_d{}_b{}_{}'.format(
            kwargs['model'],
            kwargs['data'],
            kwargs['ablation'],
            kwargs['downsample'],
            kwargs['entity_id'],
            kwargs['seq_len'],
            kwargs['pred_len'],
            kwargs['d_model'],
            kwargs['batch_size'],
            kwargs['seed'],
            )
    return setting