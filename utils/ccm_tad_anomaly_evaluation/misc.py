# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 03:21:54 2026
@author: Murad
SISLab, USF
mmurad@usf.edu
"""

import numpy as np
from sklearn.metrics import auc
from scipy.stats import iqr
from ..ts_benchmark.evaluation.metrics.affiliation.metrics import pr_from_events
from ..ts_benchmark.evaluation.metrics.affiliation.generics import convert_vector_to_events
from ..ts_benchmark.evaluation.metrics.classification_metrics_label import range_rpf


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


def get_pr_auc(recall, precision):
    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)

    area = auc(recall, precision)
    return area


def __robust_norm__(x):
    assert len(x.shape) == 2
    median, iqr_ = np.median(x, axis = 0), iqr(x, axis = 0)
    x = (x - median) / (iqr_ + 1e-9)
    # x = x.max(axis = 1)[0]
    return x


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