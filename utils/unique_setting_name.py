
def unique_name(**kwargs):
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

    return setting