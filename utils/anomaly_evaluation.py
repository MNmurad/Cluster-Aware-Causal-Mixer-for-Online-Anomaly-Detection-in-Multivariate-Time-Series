# -*- coding: utf-8 -*-
"""
Created when I should've been asleep
@author: Murad
SISLab, USF
mmurad@usf.edu
"""

import numpy as np
from argparse import Namespace
import torch
from utils.ccm_tad_anomaly_evaluation.point_based_anomaly_evaluation import anomaly_evaluation_point
from utils.ccm_tad_anomaly_evaluation.sequence_based_anomaly_evaluation import anomaly_evaluation_sequential


def anom_evaluation(args: Namespace, 
                    pred: torch.Tensor,
                    true: torch.Tensor,
                    label: np.ndarray,
                    a_type: str,
                    vali_outputs: dict | None = None,
                    detect_type: str = 'online',
                    adjust: bool = False,
                    shifting: int = 0,
                    full_anom_rslt: bool = False):
    
    assert a_type in ['pf1', 'sf1']
    selected_keys = ['TP', 'FP', 'TN', 'FN', 'TPR', 'PPV', 'FPR', 'alpha', 'F1', 'PR_AUC']
    
    ## if you want the range based and vus based metric, comment out the following line
    # selected_keys += ['VUS_ROC', 'VUS_PR', 'Aff_F1', 'Aff_P', 'Aff_R', 'Range_F1', 'Range_P', 'Range_R']
    
    aff_flag = True if (('Aff_F1' in selected_keys) and (full_anom_rslt)) else False
    range_flag = True if (('Range_F1' in selected_keys) and full_anom_rslt) else False
    vus_flag = True if (('VUS_PR' in selected_keys) and full_anom_rslt) else False
    
    if a_type == 'pf1': # point based evaluation
        aff_flag = range_flag = vus_flag = False # i don't need details metric for point-based, details metrics takes time during processing.
        anomaly_result = anomaly_evaluation_point(pred, true, label, detect_type = detect_type,
                                                  aff_flag = aff_flag, range_flag = range_flag, 
                                                  vus_flag = vus_flag)

    elif a_type == 'sf1': # Sequence based evaluation
        droplimit = 5 # called delta in the paper. default value used in our paper
        anomaly_result = anomaly_evaluation_sequential(args, pred, true, label, vali_outputs = vali_outputs, 
                                                       droplimit = droplimit, start_end_correction = True, 
                                                       aff_flag = aff_flag, range_flag = range_flag, vus_flag = vus_flag)

    subset = {k: anomaly_result[k] for k in selected_keys if k in anomaly_result}
    return subset
    
