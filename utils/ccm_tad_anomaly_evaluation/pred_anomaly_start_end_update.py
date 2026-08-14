# -*- coding: utf-8 -*-
"""
Created when I should've been asleep
@author: Murad
SISLab, USF
mmurad@usf.edu
"""

import numpy as np


def __update_anom_prediction_using_StartEndPointUpdating__(cur_pred, scores, evidence):
    # Cur_pred: prediction label
    diff_labels = np.diff(cur_pred, n = 1, axis = -1, prepend = np.zeros(1), append = np.zeros(1))
    anom_start_idx = np.where(diff_labels == 1)[0]
    anom_end_idx = np.where(diff_labels == -1)[0] - 1
    assert len(anom_start_idx) == len(anom_end_idx)
    
    anom_seq_idx = [[anom_start_idx[i], anom_end_idx[i]] for i in range(len(anom_start_idx))]
    num_anom = len(anom_seq_idx)
    
    updated_cur_pred = np.zeros(len(cur_pred), dtype = np.int32)
    for i in range(num_anom):
        start_idx = anom_seq_idx[i][0]
        scores_0_to_anomStart = np.where(scores[:start_idx + 1] == 0)[0]
        if len(scores_0_to_anomStart) == 0: # sometimes there is no zero start point for the first anomaly 
            updated_anom_start_idx = 0
        else:
            updated_anom_start_idx = scores_0_to_anomStart[-1] # taking the first one from the left
        
        try:
            end_extra = np.argmax(evidence[anom_seq_idx[i][0]:anom_seq_idx[i][1] + 1][::-1] > 0)
        except:
            pass
        updated_anom_end_idx = anom_seq_idx[i][1] - end_extra
        updated_cur_pred[updated_anom_start_idx: updated_anom_end_idx + 1] = 1
        
    return updated_cur_pred