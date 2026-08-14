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
from .pred_anomaly_start_end_update import __update_anom_prediction_using_StartEndPointUpdating__
from .misc import affiliation_scores, range_based_metrics, get_pr_auc, get_f1
from sklearn.mixture import GaussianMixture


def anomaly_evaluation_sequential(args, pred, true, label, vali_outputs = None, droplimit = None, 
                                  full_anom_rslt = False, start_end_correction = None,
                                  aff_flag = False, range_flag = False, vus_flag = False):
    assert isinstance(label, np.ndarray)

    epsilon = 1e-8
    bst_f1 = -np.inf  
    
    vali_score = vali_outputs['nominal_score']
    pred = pred.numpy().squeeze() if torch.is_tensor(pred) else pred.squeeze()
    true = true.numpy().squeeze() if torch.is_tensor(true) else true.squeeze()
    label = label.astype(int)    
    
    test_score = ((pred - true) ** 2).mean(1)
    vali_score_sorted = np.sort(vali_score)
    
    num_greater_test = len(vali_score_sorted) - np.searchsorted(vali_score_sorted, test_score, side='right')
    p_score = (num_greater_test / len(vali_score_sorted)) + epsilon # original

    alphax = get_top_best_threshold(test_score, label, vali_score_sorted, 10, iter_steps = 1000)
    additional_alpha = [0.0001, 0.0002, 0.005, 0.05, 0.1, 0.2, 0.5] # **additional** alpha to search.
    alphas = list(alphax) + additional_alpha
    alphas.sort()
    
    # # # If you want our proposed heuristic range selection of alpha, then comment_out the following
    # if args.data in ['SWAT', 'WADI', 'PSM']:
    #     mx_alphas = multi_gaussian(vali_score, 1)
    #     alphas = [a for a in alphas if a < mx_alphas]

    
    bst_results = None
    aff_f1 = [-np.inf]; aff_p = [-np.inf]; aff_r = [-np.inf]
    range_f1 = [-np.inf]; range_p = [-np.inf]; range_r = [-np.inf]
    
    for alp in alphas:
        beta_t = np.log(alp / p_score) # evidence: beta_t in paper
        s_t = cumsum_fast(beta_t, droplimit) # score: s_t in paper
        results = __SequenceBasedResults__(s_t, label, beta_t, droplimit = droplimit, start_end_correction = start_end_correction, 
                                           adjust = False, iter_steps = 1000, aff_flag = aff_flag, range_flag = range_flag)
        
        if results['F1'] == -np.inf:
            continue

        if results['F1'] > bst_f1:
            bst_results = results 
            bst_f1 = results['F1']
            bst_results['alpha'] = alp
            beta_t_optim = beta_t
            s_t_optim = s_t
        
        if results.get('Aff_F1', -np.inf) > aff_f1[-1]:
            aff_f1.append(results['Aff_F1'])
            aff_p.append(results['Aff_P'])
            aff_r.append(results['Aff_R'])
        
        if results.get('Range_F1', -np.inf) > range_f1[-1]:
            range_f1.append(results['Range_F1'])
            range_p.append(results['Range_P'])
            range_r.append(results['Range_R'])
            
    # VUS result for optimum alpha
    if vus_flag:
        _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(label, s_t_optim, 100, 
                                                           __update_anom_prediction_using_StartEndPointUpdating__, 
                                                           evidence = beta_t_optim, start_end_update = start_end_correction, version='opt_mem', thre = 1000)
    else:
        VUS_ROC, VUS_PR = 0, 0
        
    bst_results['VUS_ROC'] = VUS_ROC
    bst_results['VUS_PR'] = VUS_PR
    
    # if i wnat aff scores from all max posibilities; otherwise it would be for the similar alpha for best F1
    if bst_results.get('Aff_F1'):
        bst_results['Aff_F1'] = aff_f1[np.argmax(aff_f1)]
        bst_results['Aff_P'] = aff_p[np.argmax(aff_f1)]
        bst_results['Aff_R'] = aff_r[np.argmax(aff_f1)]
        
    # if I wnat range scores from all max posibi.; otherwise it would be for the similar alpha for best F1
    if bst_results.get('Range_F1'):
        bst_results['Range_F1'] = range_f1[np.argmax(range_f1)]
        bst_results['Range_P'] = range_p[np.argmax(range_f1)]
        bst_results['Range_R'] = range_r[np.argmax(range_f1)]
    return bst_results


def __SequenceBasedResults__(scores, true_label, evidence, droplimit = None, 
                             start_end_correction = None, iter_steps=1000, adjust = False, min_ratio=0,
                             aff_flag = False, range_flag = False):
    """
    Get threshold of anomaly scores corresponding to the best f1.
    
    scores: s_t in paper
    evidence: beta_t in paper
    
    Returns:
        - threshold: Best threshold.
        - f1: best f1 score.
    """

    total_anomaly_pts = true_label.sum() # number of anomaly points
    total_normal_pts = len(true_label) - total_anomaly_pts # number of normal points

    ######
    q = np.linspace(0, 1, iter_steps)
    unique_score_values = np.unique(scores)
    threshold = np.quantile(unique_score_values, q)
    nthreshold = len(threshold)
    ######
    
    if nthreshold == 1: # all the score values are fixed constant. No meaning to check
        return {'F1': -np.inf}
    
    scores_2d_matrix = np.repeat(scores[None, :], nthreshold, axis = 0) # [nthreshold, L]
    prediction = scores_2d_matrix >= threshold[:, None] # [nthreshold, L]    
    
    # updating anomaly start and end point using our proposed method
    prev_cur_pred = []
    updated_prediction = []
    Aff_F1 = -np.inf
    R_F1 = -np.inf
    additional_results = {}
    
    for i, cur_pred in enumerate(prediction):
        if start_end_correction:
            updated_prediction.append(__update_anom_prediction_using_StartEndPointUpdating__(cur_pred, scores, evidence)) 
        else: 
            updated_prediction.append(cur_pred)
            
        ########### To make the aff_score calculation faster, i will only consider the prediction, when it changes compared to previous thershold
        # if previous pred and current pred are same, no need to calculate aff twice.
        if (i == 0):
            change = 1
        else:
            change = (updated_prediction[-1] - prev_cur_pred).sum()
        prev_cur_pred = updated_prediction[-1]
        if change:
            aff_results = affiliation_scores(true_label, updated_prediction[-1]) if aff_flag else {}
            range_results = range_based_metrics(true_label, updated_prediction[-1]) if range_flag else {}
            if aff_results.get('Aff_F1', -np.inf) > Aff_F1:
                Aff_F1 = aff_results['Aff_F1']
                additional_results = additional_results | aff_results
            if range_results.get('Range_F1', -np.inf) > R_F1:
                R_F1 = range_results['Range_F1']
                additional_results = additional_results | range_results
                
    ############ 
    
    updated_prediction = np.stack(updated_prediction) # [nthreshold, L]
    
    labels_2d = true_label[None, :] # [1, L]
    TP = (updated_prediction * labels_2d).sum(-1) # [nthreshold]
    FP = updated_prediction.sum(-1) - TP # [nthreshold]
    TN = total_normal_pts - FP # [nthreshold]
    FN = total_anomaly_pts - TP # [nthreshold]
    
    # TPR, FPR, F1
    TPR = TP / total_anomaly_pts # Recall: detected true pos / Number of actual true pos ( TPs + FNs) = total_anomaly_pts.
    FPR = FP / total_normal_pts # False Positives (FP)​ / Total Negatives
    Precision = TP / (TP + FP) # Precision
    Recall = TPR
    
    # Precision and Recall curve should start from precision = 1, recall 0
    Recall = np.append(Recall, 0.0)
    Precision = np.append(Precision, 1.0)
    TPR = np.append(TPR, 0)
    FPR = np.append(FPR, 0)
    
    F1  = 2 * Recall * Precision / (Recall + Precision + 1e-16) 
    try:
        pr_auc_value  = get_pr_auc(Recall, Precision)
    except:
        pass
    # Best F1 Related Results
    bst_F1_idx = F1.argmax()
    
    result = {'F1': F1[bst_F1_idx],
              'thres': threshold[bst_F1_idx], 
              'TP': TP[bst_F1_idx],
              'FP': FP[bst_F1_idx],
              'TN': TN[bst_F1_idx],
              'FN': FN[bst_F1_idx],
              'TPR': TPR[bst_F1_idx], 
              'PPV': Precision[bst_F1_idx], 
              'FPR': FPR[bst_F1_idx],
              'maxid': bst_F1_idx, 
              'PR_AUC': pr_auc_value,
              'pred_label': updated_prediction[bst_F1_idx, :],
              'Precision_values': Precision,
              'Recall_values': Recall,
              'threshold_list': threshold, # required for sensitivity analysis
              'F1_list': F1[:-1] # required for sensitivity analysis, last one is ignored, because it is related to the last appended recall and precision value
              }
    
    result = result | additional_results
    return result


def cumsum_fast(scores, droplimit):
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    out = np.empty(n, dtype=float)
    prev = 0.0
    count = 0

    for i in range(n):
        s = scores[i]
        if s > 0:
            count = 0
        else:
            count += 1
            if count > droplimit:
                prev = 0.0

        tmp = prev + s
        prev = tmp if tmp > 0 else 0.0
        out[i] = prev

    return out



def get_top_best_threshold(scores, true, vali_score_sorted, k, iter_steps = 1000):

    sorted_index = np.argsort(scores)
    th_vals = np.linspace(0, len(scores) - 1, num=iter_steps)
    # k = 10
    best_f1, best_thr = 0, 0
    f1_list, th_list = [], []
    
    for th_idx in th_vals:
        cur_thr = scores[sorted_index[int(th_idx)]]
        cur_pred = (scores >= cur_thr).astype(int)
        cur_f1 = get_f1(cur_pred, true)
        
        f1_list.append(cur_f1)
        th_list.append(cur_thr)
        
        if cur_f1 > best_f1:
            best_f1, best_thr = cur_f1, cur_thr
            
    f1_array = np.asarray(f1_list)
    th_array = np.asarray(th_list)
    
    possible_optim_point = np.where(f1_array == f1_array.max())[0][0]
    max_pop = min(possible_optim_point + k, len(f1_array))
    min_pop = max(possible_optim_point - k, 0)
    
    selected_thr = th_array[min_pop:max_pop]

    num_greater_test = len(vali_score_sorted) - np.searchsorted(vali_score_sorted, selected_thr, side = 'right')
    p_score = (num_greater_test / len(vali_score_sorted)) + 1e-9
    
    return p_score


def multi_gaussian(vali_score, nstd):
    X = vali_score.reshape(-1, 1)
    bics = []
    Ks = range(1, 6)

    for k in Ks:
        gmm = GaussianMixture(n_components=k, random_state=0)
        gmm.fit(X)
        bics.append(gmm.bic(X))

    best_k = Ks[np.argmin(bics)]
    # print("Best number of components:", best_k)

    gmm = GaussianMixture(n_components=best_k, random_state=0)
    gmm.fit(X)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_

    idx = np.argmin(means) # np.argmax(weights)
    mean = means[idx: idx + 1]
    std = stds[idx: idx + 1]
    weight = weights[idx: idx + 1]
    
    lower_limit = mean + nstd * std
    max_alpha = 1 - (np.sum(vali_score < (lower_limit)) / len(vali_score))
    max_alpha = 1e-6 if max_alpha == 0 else max_alpha
    
    return max_alpha


# def get_top_best_threshold(scores, true, vali_score_sorted, k, iter_steps = 1000):
#     q = np.linspace(0, 1, iter_steps)
#     unique_score_values = np.unique(scores)
#     thresholds = np.quantile(unique_score_values, q)
    
#     best_f1 = 0
    
#     for i, cur_thr in enumerate(thresholds):
#         cur_pred = (scores >= cur_thr).astype(int)
#         cur_f1 = get_f1(cur_pred, true)

#         if cur_f1 > best_f1:
#             best_f1 = cur_f1
#             best_idx = i
            
#     num_greater_test = len(vali_score_sorted) - np.searchsorted(vali_score_sorted, thresholds, side = 'right')
#     p_score = (num_greater_test / len(vali_score_sorted)) + 1e-9
#     optimal_p_score = p_score[best_idx]
    
#     unique_p_score = np.unique(p_score)
#     m = np.where(unique_p_score == optimal_p_score)[0][0]
#     m1 = max(m - k, 0)
#     m2 = min(m1 + 2*k, len(unique_p_score))
    
#     out_p_score = unique_p_score[m1:m2]
#     return out_p_score

