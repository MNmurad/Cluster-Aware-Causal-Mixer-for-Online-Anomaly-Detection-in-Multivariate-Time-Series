
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
from scipy.stats import iqr
# import pdb
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import percentileofscore
from sklearn.metrics import precision_recall_curve, auc
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import time
from sklearn.mixture import GaussianMixture
from collections import defaultdict
from .ts_benchmark.evaluation.metrics.affiliation.metrics import pr_from_events
from .ts_benchmark.evaluation.metrics.affiliation.generics import convert_vector_to_events
from .ts_benchmark.evaluation.metrics.classification_metrics_label import range_rpf


def anom_evaluation(args, pred, true, label, a_type, vali_outputs = None, detect_type = 'online', adjust = False, shifting = 0, full_anom_rslt = False):
    assert a_type in ['pf1', 'sf1']
    selected_keys = ['TP', 'FP', 'TN', 'FN', 'TPR', 'PPV', 'FPR', 'alpha', 'F1', 
                     'PR_AUC', 
                     'VUS_ROC', 'VUS_PR', 
                     'Aff_F1', 'Aff_P', 'Aff_R',
                     'Range_F1', 'Range_P', 'Range_R',
                     ]
    
    aff_flag = True if (('Aff_F1' in selected_keys) and (full_anom_rslt)) else False
    range_flag = True if (('Range_F1' in selected_keys) and full_anom_rslt) else False
    vus_flag = True if (('VUS_PR' in selected_keys) and full_anom_rslt) else False
    

    if a_type == 'pf1':
        if args.model == 'Basic_Mixer': # for our model, i don't need details metric for point-based, details metrics takes time during processing.
            aff_flag = range_flag = vus_flag = False
            
        anomaly_result = anomaly_evaluation_point(pred, true, label, detect_type = detect_type,
                                                  aff_flag = aff_flag, range_flag = range_flag, vus_flag = vus_flag)


    elif a_type == 'sf1':
        droplimit = 5
        anomaly_result = anomaly_evaluation_sequential(args, pred, true, label, vali_outputs = vali_outputs, droplimit = droplimit, start_end_correction = True, 
                                                       aff_flag = aff_flag, range_flag = range_flag, vus_flag = vus_flag)

     
    subset = {k: anomaly_result[k] for k in selected_keys if k in anomaly_result}
    return subset
    

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


from utils.vus.metrics import generate_curve
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
    results['alpha'] = 0
    
    return results



def anomaly_evaluation_sequential(args, pred, true, label, vali_outputs = None, droplimit = None, 
                                  full_anom_rslt = False, start_end_correction = None,
                                  aff_flag = False, range_flag = False, vus_flag = False):
    assert isinstance(label, np.ndarray)
    # # Sensitivity analysis plot
    # __sensitivity_analysis_sequential_method__(pred, true, label, vali_outputs = None, droplimit = None, full_anom_rslt = False, start_end_correction = None)
    
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
    
    # If you want our proposed heuristic range selection of alpha, then comment_out the following
    if args.data in ['SWAT', 'WADI', 'PSM']:
        mx_alphas = multi_gaussian(vali_score, 1)
        alphas = [a for a in alphas if a < mx_alphas]

    
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
        _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(label, s_t_optim, 100, evidence = beta_t_optim, start_end_update = start_end_correction, version='opt_mem', thre = 1000)
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


def cumsum(scores, droplimit): # et= scores, drop: delta
    n = len(scores)
    count = 0
    out = [0] # s
    for i in range(n):
        if scores[i] > 0:
            prev = out[-1] # 
            count = 0
        elif scores[i] <= 0:
            count += 1
            if count > droplimit:
                prev = 0
            else:
                prev = out[-1]
                
        out.append(max(prev + scores[i], 0))
    return np.asarray(out[1:])
#############################################

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





def getCombinedResult(results):
    n = len(results)
    
    combined = {key: [d[key] for d in results] for key in results[0]}
    averaged = {key: sum(d[key] for d in results) / len(results) for key in results[0] if key != 'alpha'}
    summed = {key: sum(d[key] for d in results) for key in results[0] if key != 'alpha'}
    final_output = {}
    
    # Concatenated F1 score:
    TPs = summed['TP']
    FPs = summed['FP']
    TNs = summed['TN']
    FNs = summed['FN']
    precision = TPs / (TPs + FPs) if (TPs + FPs) != 0 else 0
    recall = TPs / (TPs + FNs) if (TPs + FNs) != 0 else 0
    # this is similar to when I concate pred label of all entities, then calculate the f1
    f1_concat = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0 
    
    final_output = {'avg' + key: averaged[key] for key in averaged.keys()}
    final_output['alphas'] = combined['alpha']
    final_output['f1_concat'] = f1_concat
    final_output['TP_concat'] = TPs
    final_output['FP_concat'] = FPs
    final_output['TN_concat'] = TNs
    final_output['FN_concat'] = FNs
    return final_output


# def getCombinedResult(results):
#     n = len(results)
#     TP, FP, TN, FN, F1 = [], [], [], [], [] # used to accumulate the values
#     PR_AUC = []
#     VUS_ROC, VUS_PR = [], []
#     Aff_F1, Aff_P, Aff_R = [], [], []
#     gflops = []
#     alpha = []
#     nParams, infTime = [], []
#     for i in range(0, n):
#         TP.append(results[i]['TP'])
#         FP.append(results[i]['FP'])
#         TN.append(results[i]['TN'])
#         FN.append(results[i]['FN'])
#         F1.append(results[i]['F1'])
#         PR_AUC.append(results[i]['PR_AUC'])
#         VUS_ROC.append(results[i]['VUS_ROC'])
#         VUS_PR.append(results[i]['VUS_PR'])
        
#         if results[i].get('Aff_F1'): # if provided
#             Aff_F1.append(results[i]['Aff_F1'])
#             Aff_P.append(results[i]['Aff_P'])
#             Aff_R.append(results[i]['Aff_R'])
            
#         gflops.append(results[i]['gflops'])
#         nParams.append(results[i]['nParams'])
#         infTime.append(results[i]['avgInfTime'])
#         try:
#             alpha.append(results[i]['alpha'])
#         except:
#             pass
    
#     TPs = np.sum(TP)
#     FPs = np.sum(FP)
#     TNs = np.sum(TN)
#     FNs = np.sum(FN)
#     precision = TPs / (TPs + FPs) if (TPs + FPs) != 0 else 0
#     recall = TPs / (TPs + FNs) if (TPs + FNs) != 0 else 0
    
#     # this is similar to when I concate pred label of all entities, then calculate the f1
#     f1_concat = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0 
#     # this is avg of f1 from all entities
#     avgF1 = np.mean(F1)
#     avgGFLOPs = np.mean(gflops)
#     avgPR_AUC = np.mean(PR_AUC)
#     avgVUS_ROC = np.mean(VUS_ROC)
#     avgVUS_PR = np.mean(VUS_PR)
#     avgnParams = np.mean(nParams)
#     avginfTime = np.mean(infTime)
    
#     additional_results = {}
#     if len(Aff_F1) == len(F1): # similar to if AFF_F1 provided.....
#         avgAff_F1 = np.mean(Aff_F1)
#         avgAff_P = np.mean(Aff_P)
#         avgAff_R = np.mean(Aff_R)
#         additional_results = {'avgAff_F1': avgAff_F1, 'avgAff_P': avgAff_P, 'avgAff_R': avgAff_R}
    
#     out = {'alphas': alpha, 'avgF1': avgF1, 'f1_concat': f1_concat, 
#            'TPs': TPs, 'FPs': FPs, 'TNs': TNs, 'FNs': FNs, 
#             'avgPR_AUC': avgPR_AUC, 'avgVUS_ROC': avgVUS_ROC, 'avgVUS_PR': avgVUS_PR,
#             'avgGFLOPs': avgGFLOPs, 'avgnParams': avgnParams, 'avginfTime': avginfTime}
    
#     out = out|additional_results
    
#     return out, F1


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



def __sensitivity_analysis_sequential_method__(pred, true, label, vali_outputs = None, droplimit = None, full_anom_rslt = False, start_end_correction = None):
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
    p_score = (num_greater_test / len(vali_score_sorted)) + epsilon
    
    topk = 1000
    alphax = get_top_best_threshold(test_score, label, vali_score_sorted, topk, iter_steps = 1000)
    additional_alpha = [0.0001, 0.0002, 0.005, 0.05, 0.1, 0.2, 0.5] # **additional** alpha to search.
    
    bst_results = None
    results_alpha_dic = {'alpha': [], 'Precision_stack': [], 'Recall_stack': [], 'PR_AUC': [], 'F1': [], 'F1_list': [], 'threshold_list': []}
    
    for alp in list(alphax) + additional_alpha:
        beta_t = np.log(alp / p_score)
        # s_t = cumsum(final_scores, droplimit)
        s_t = cumsum_fast(beta_t, droplimit)
        results = __SequenceBasedResults__(s_t, label, beta_t, droplimit = droplimit, start_end_correction = start_end_correction, adjust = False, iter_steps = 1000)
        
        if results['F1'] == -np.inf:
            continue
        
        results_alpha_dic['alpha'].append(alp) # list
        results_alpha_dic['Precision_stack'].append(results['Precision_values'])
        results_alpha_dic['Recall_stack'].append(results['Recall_values'])
        results_alpha_dic['PR_AUC'].append(results['PR_AUC']) # list
        results_alpha_dic['F1'].append(results['F1']) # list
        results_alpha_dic['F1_list'].append(results['F1_list']) # list of list: For each alpha, I have a list of threshold and their corresponding F1values
        results_alpha_dic['threshold_list'].append(results['threshold_list']) # list of list: For each alpha, I have a list of threshold and their corresponding F1values

        if results['F1'] > bst_f1:
            bst_results = results 
            bst_f1 = results['F1']
            bst_results['alpha'] = alp
    

    # ####### This part is to plot the sensitivity of alpha ###################
    alpha_list = results_alpha_dic['alpha']
    f1_list = results_alpha_dic['F1']
    pr_auc_list = results_alpha_dic['PR_AUC']
    
    # sort by alpha
    alpha_list, f1_list, pr_auc_list = zip(
        *sorted(zip(alpha_list, f1_list, pr_auc_list), key=lambda x: x[0])
    )
    
    alpha_list = np.array(alpha_list)
    f1_list = np.array(f1_list)
    pr_auc_list = np.array(pr_auc_list)
    
    # limit alpha <= 0.9
    mask = alpha_list <= 0.9
    alpha_list = alpha_list[mask]
    f1_list = f1_list[mask]
    pr_auc_list = pr_auc_list[mask]
    
    # ALPHA vs F1
    plt.figure(figsize=(4, 2.5), dpi = 600)
    plt.grid(True, which = 'both', linestyle='--', linewidth=0.6, alpha = 0.6)
    plt.margins(x=0, y=0)
    plt.plot(alpha_list, f1_list, linewidth = 2, color = 'blue')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('F1 Score')
    plt.tight_layout(pad = 0.2)
    plt.show()
    plt.savefig('alpha_f1.png', bbox_inches='tight',pad_inches=0.02)
    
    # h vs F1
    plt.figure(figsize=(4, 2.5), dpi = 600)
    plt.grid(True, which = 'both', linestyle='--', linewidth=0.6, alpha = 0.6)
    plt.margins(x=0, y=0)
    plt.plot(bst_results['threshold_list'], bst_results['F1_list'], linewidth=2, color = 'blue')
    plt.xscale('log')
    plt.text(0.98, 0.95, rf'$\alpha = {bst_results["alpha"]:.4f}$', transform=plt.gca().transAxes, ha='right', va='top', fontsize=11)
    plt.xlabel(r'$h$')
    plt.ylabel('F1 Score')
    plt.tight_layout(pad = 0.2)
    plt.show()
    plt.savefig('h_f1.png', bbox_inches='tight',pad_inches=0.02)
    
    # ALPHA vs PRAUC
    plt.figure(figsize=(4, 2.5), dpi = 600)
    plt.grid(True, which = 'both', linestyle='--', linewidth=0.6, alpha = 0.6)
    plt.margins(x=0, y=0)
    plt.plot(alpha_list, pr_auc_list, linewidth=2, color = 'blue')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('PR-AUC')
    plt.tight_layout(pad = 0.2)
    plt.show()
    plt.savefig('alpha_pr-auc.png', bbox_inches='tight',pad_inches=0.02)
    
    
    # ####### Sensitivity of joint alpha and h ###################################
    alpha_list2 = []
    h_list2 = []
    f1_list2 = []
    
    for a, h, f in zip(results_alpha_dic['alpha'], results_alpha_dic['threshold_list'], results_alpha_dic['F1_list']):
        alpha_list2.append([a]*len(h))
        
    alpha_list2 = np.concatenate(alpha_list2)
    h_list2 = np.concatenate(results_alpha_dic['threshold_list'])
    f1_list2 = np.concatenate(results_alpha_dic['F1_list'])
    assert len(f1_list2) == len(h_list2) == len(alpha_list2)
    
    # create grid
    # alpha_grid = np.linspace(alpha_list2.min(), alpha_list2.max(), 1000)
    # h_grid = np.linspace(h_list2.min(), h_list2.max(), 1000)
    # A, H = np.meshgrid(alpha_grid, h_grid)
    alpha_grid = np.sort(np.unique(alpha_list2))
    hgrid_min = h_list2.min()
    hgrid_max = 500 # h_list2.max()
    h_grid = np.linspace(hgrid_min, hgrid_max, 1000) # hmax = 500, points 1000
    A, H = np.meshgrid(alpha_grid, h_grid)
    
    # interpolate F1 onto grid
    F1_grid = griddata(
        (alpha_list2, h_list2),
        f1_list2,
        (A, H),
        method='linear', # 'linear'
    )

    plt.figure(figsize=(4, 2.5), dpi = 600)
    im = plt.imshow(
        F1_grid,
        origin='lower',
        aspect='auto',
        extent=[alpha_grid.min(), alpha_grid.max(), h_grid.min(), h_grid.max()]
    )
    
    plt.colorbar(im, label='F1 Score')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$h$')
    # plt.title('F1 Score over $(\\alpha, h)$')
    
    plt.tight_layout(pad = 0.2)
    plt.show()
    plt.savefig('Join_alpha_h.png', bbox_inches='tight',pad_inches=0.02)
    return None



def get_top_best_threshold(scores, true, vali_score_sorted, k, iter_steps = 1000):
    q = np.linspace(0, 1, iter_steps)
    unique_score_values = np.unique(scores)
    thresholds = np.quantile(unique_score_values, q)
    
    best_f1 = 0
    
    for i, cur_thr in enumerate(thresholds):
        cur_pred = (scores >= cur_thr).astype(int)
        cur_f1 = get_f1(cur_pred, true)

        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_idx = i
            
    num_greater_test = len(vali_score_sorted) - np.searchsorted(vali_score_sorted, thresholds, side = 'right')
    p_score = (num_greater_test / len(vali_score_sorted)) + 1e-9
    optimal_p_score = p_score[best_idx]
    
    unique_p_score = np.unique(p_score)
    m = np.where(unique_p_score == optimal_p_score)[0][0]
    m1 = max(m - k, 0)
    m2 = min(m1 + 2*k, len(unique_p_score))
    
    out_p_score = unique_p_score[m1:m2]
    return out_p_score


def __robust_norm__(x):
    assert len(x.shape) == 2
    median, iqr_ = np.median(x, axis = 0), iqr(x, axis = 0)
    x = (x - median) / (iqr_ + 1e-9)
    # x = x.max(axis = 1)[0]
    return x


def get_precision(pred, true):
    correct_num = (pred & true).sum()
    return correct_num / (pred.sum() + 1e-8)


def get_recall(pred, true):
    correct_num = (pred & true).sum()
    return correct_num / (true.sum() + 1e-8)


def get_f1(pred, true):
    precision = get_precision(pred, true)
    recall = get_recall(pred, true)
    return 2 * precision * recall / (precision + recall + 1e-8)




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
        _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(labels, scores, 100, evidence = None, start_end_update = False, version='opt_mem', thre = 1000)
    else:
        VUS_ROC, VUS_PR = 0, 0
    
    # ################################ additional metric 
    q = np.linspace(0, 1, 500)
    unique_threshold_values = np.unique(threshold_list)
    selected_unique_threshold_values = np.quantile(unique_threshold_values, q)
    
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



def get_pr_auc(recall, precision):
    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)


    area = auc(recall, precision)
    return area



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

    # for i in range(best_k):
    #     print(f"Component {i}: mean={means[i]:.6f}, std={stds[i]:.6f}, weight={weights[i]:.3f}")
    
    idx = np.argmin(means) # np.argmax(weights)
    mean = means[idx: idx + 1]
    std = stds[idx: idx + 1]
    weight = weights[idx: idx + 1]
    
    lower_limit = mean + nstd * std
    max_alpha = 1 - (np.sum(vali_score < (lower_limit)) / len(vali_score))
    max_alpha = 1e-6 if max_alpha == 0 else max_alpha
    
    return max_alpha


def affiliation_scores(y_test_label, y_pred_label):
    """ Taken from ... paper"""
    events_gt = convert_vector_to_events(y_test_label) # [(4, 5), (8, 9)]
    events_pred = convert_vector_to_events(y_pred_label)     # [(3, 4), (7, 10)]
    Trange = (0, len(y_test_label))
    affiliation = pr_from_events(events_pred, events_gt, Trange)
    aff_p = affiliation['precision']
    aff_r = affiliation['recall']
    aff_f1 = 2 * aff_p * aff_r / (aff_p + aff_r) if (aff_p + aff_r) > 0 else 0.0
    return {'Aff_F1': aff_f1, 'Aff_P': aff_p, 'Aff_R': aff_r} # {'Aff_F1': aff_f1, 'Aff_P': aff_p, 'Aff_R': aff_r}

def range_based_metrics(y_test_label, y_pred_label):
    RPF = range_rpf(y_test_label, y_pred_label)
    return {'Range_R': RPF[0], 'Range_P': RPF[1], 'Range_F1': RPF[2]} # RPF # recall, preci, f1
