
import numpy as np
import torch

def __process_cluster_info__(clusterid, nheads, channel, d_model):
    '''
    cluster id for each channels: [cluster-x, cluster-y, cluster-z.....]
    '''
    # ########## Testing Params ############
    # clusterid = [0, 3, 0, 3, 1, 0, 0, 0, 3, 3, 1, 1, 3, 1, 3, 1, 3, 3, 2, 0, 3, 0, 0, 0, 3, 0, 0, 2,
    #              1, 1, 2, 1, 1, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 1, 2, 0, 2, 0, 1, 0, 1]
    # nheads = 4
    # d_model = 128
    # channel = 51
    # ####################################
    
    ngroups = len(np.unique(clusterid)) # number of clusters
    groupChannels = [] # list: [[channels in cluster-1], [channels in cluster-2],........]
    cid = np.array(clusterid)
    assert (nheads == ngroups) and (nheads <= channel)
    
    c = [] # list: [number_of_channles_in_cluster_1, number_of_channles_in_cluster_2, ...........]
    d = [] # list: [part_of_dmodel_alloted_to_cluster_1, part_of_dmodel_alloted_to_cluster_2, ...........]
    for i in range(ngroups):
        groupChannels.append(list(np.where(cid == i)[0])) # list of the channels in cluster-i.
        c.append(len(groupChannels[i]))
        d.append(int((len(groupChannels[i]) / channel) * d_model))
    d[-1] += d_model - sum(d) # Ensuring the sum of all d is equal to d_model
    
    ''' Relation among c, d, groupChannels:
        c[i]: number of channels in cluster-i
        d[i]: allocated d_model portion to cluster-i
        groupChannels[i]: list of the channels inside cluster-i
    '''
    channels_after_clusterring = [c for cgs in groupChannels for c in cgs] # all channels id are merged according to cluster
    channels_recover_position = np.argsort(channels_after_clusterring) # this will be used to recover the channels in their original chronological position
    
    out = {'c': c, 
           'd': d,
           'groupChannels': groupChannels, 
           'cac': channels_after_clusterring,
           'crp': channels_recover_position}
    
    return out
    