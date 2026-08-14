# -*- coding: utf-8 -*-
"""
Created when I should've been asleep
@author: Murad
SISLab, USF
mmurad@usf.edu
"""


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