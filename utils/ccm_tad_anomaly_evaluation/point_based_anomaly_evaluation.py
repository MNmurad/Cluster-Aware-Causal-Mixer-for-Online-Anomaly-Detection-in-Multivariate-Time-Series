# -*- coding: utf-8 -*-
"""
Created when I should've been asleep
@author: Murad
SISLab, USF
mmurad@usf.edu
"""

import numpy as np
import torch
from utils.vus.metrics import generate_curve # position of this module calling is important
from utils.ccm_tad_anomaly_evaluation.misc import get_pr_auc
from utils.ccm_tad_anomaly_evaluation.misc import affiliation_scores
from utils.ccm_tad_anomaly_evaluation.misc import range_based_metrics
from utils.ccm_tad_anomaly_evaluation.misc import __robust_norm__


def anomaly_evaluation_point(pred, true, label, detect_type = 'online', aff_flag = False, range_flag = False, vus_flag = False):
    assert isinstance(label, np.ndarray)
    
    pred = pred.numpy().squeeze() if torch.is_tensor(pred) else pred.squeeze()
    true = true.numpy().squeeze() if torch.is_tensor(true) else true.squeeze()
    label = label.astype(int)    

    if detect_type == 'online':
        scores_online = ((pred - true) ** 2).mean(1)
        final_scores = scores_online
        
    if detect_type == 'offline':
        scores_offline = ((pred - true) ** 2)
        scores_off_norm = __robust_norm__(scores_offline)
        final_scores = scores_off_norm.max(axis = 1)
    
    results = __PointBasedResults__(scores = final_scores, labels = label, aff_flag = aff_flag, range_flag = range_flag, vus_flag = vus_flag)
    results['alpha'] = 0 # no meaning for point based evaluation
    return results


def __PointBasedResults__(scores = None, labels = None, aff_flag = False, range_flag = False, vus_flag = False):
    scores = scores.numpy() if torch.is_tensor(scores) else scores
    labels = labels.numpy() if torch.is_tensor(labels) else labels
    total_anomaly_pts = labels.sum() # number of anomaly points
    total_normal_pts = len(labels) - total_anomaly_pts # number of normal points
    
    sortid = np.argsort(scores)
    new_label = labels[sortid]
    threshold_list = scores[sortid] # small to large
    
    TP = np.cumsum(-new_label) + total_anomaly_pts    
    FP = np.cumsum(new_label-1) + total_normal_pts # zeros could be considered as max number of False Positive.
    FN = total_anomaly_pts - TP
    TN = total_normal_pts - FP
    
    # N = len(labels) - np.flip(TPs > 0).argmax() # Due to cancelling the bottom portion, where there is no anomaly?
    N = -1
    TPR = TP[:N] / total_anomaly_pts # Recall: detected true pos / Number of actual true pos ( TPs + FNs) = ones.
    FPR = FP[:N] / total_normal_pts # False Positives (FP)​ / Total Negatives
    Precision = TP[:N] / (TP + FP)[:N] # Precision
    Recall = TPR

    # Precision and Recall curve should start from precision = 1, recall 0
    Recall = np.append(Recall, 0.0)
    Precision = np.append(Precision, 1.0)
    TPR = np.append(TPR, 0)
    FPR = np.append(FPR, 0)
    
    F1  = 2 * Recall * Precision / (Recall + Precision + 1e-16) 
    pr_auc_value  = get_pr_auc(Recall, Precision)
    
    # Best F1 Related Results
    bst_F1_idx = F1.argmax()
    prediction = scores >= threshold_list[bst_F1_idx]
    
    if vus_flag:
        _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(labels, scores, 100, 
                                                           function_PredAnomStartEnd_update = None, 
                                                           evidence = None, start_end_update = False, version='opt_mem', thre = 1000)
    else:
        VUS_ROC, VUS_PR = 0, 0
    
    # ################################ additional metric 
    q = np.linspace(0, 1, 500)
    unique_threshold_values = np.unique(threshold_list)
    selected_unique_threshold_values = np.quantile(unique_threshold_values, q)
    
    prev_cur_pred = []
    Aff_F1 = -np.inf
    R_F1 = -np.inf
    additional_results = {}
    if aff_flag or range_flag:
        for i, cur_thr in enumerate(selected_unique_threshold_values):
            cur_pred = (scores >= cur_thr).astype(np.int32)
            ########### To make the aff_score calculation faster, i will only consider the prediction, when it changes compared to previous thershold
            # if previous pred and current pred are same, no need to calculate aff twice.
            if (i == 0):
                change = 1
            else:
                change = (cur_pred - prev_cur_pred).sum()
            prev_cur_pred = cur_pred
            if change:
                aff_results = affiliation_scores(labels, cur_pred) if aff_flag else {}
                range_results = range_based_metrics(labels, cur_pred) if range_flag else {}
                if aff_results.get('Aff_F1', -np.inf) > Aff_F1:
                    Aff_F1 = aff_results['Aff_F1']
                    additional_results = additional_results | aff_results
                if range_results.get('Range_F1', -np.inf) > R_F1:
                    R_F1 = range_results['Range_F1']
                    additional_results = additional_results | range_results
    ########################################
    
    result = {'F1': F1[bst_F1_idx],
              'thres': threshold_list[bst_F1_idx], 
              'TP': TP[bst_F1_idx],
              'FP': FP[bst_F1_idx],
              'TN': TN[bst_F1_idx],
              'FN': FN[bst_F1_idx],
              'TPR': TPR[bst_F1_idx], 
              'PPV': Precision[bst_F1_idx], 
              'FPR': FPR[bst_F1_idx],
              'maxid': bst_F1_idx, 
              'PR_AUC': pr_auc_value,
              'pred_label': prediction,
              'VUS_ROC': VUS_ROC,
              'VUS_PR': VUS_PR
              }
    
    result = result | additional_results
    return result

