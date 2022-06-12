from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np

def retrieval(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind) * 100
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind) * 100
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind) * 100
    metrics['MR'] = np.median(ind) + 1
    return metrics

def ctr(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    # ind = np.where(ind == 0)
    # ind = [(i, j) for i, j in zip(ind[0], ind[1])]
    
    # new_ind = []
    # for i in ind:
    #     ind_set = set([j[0] for j in new_ind])
    #     if i[0] not in ind_set:
    #         new_ind.append(i)
    # ind = np.array([i[1] for i in new_ind])
    
    num = 0.
    count = 0.
    for i in ind:
        if i[0] == 0:
            num += 1
        count += 1
    
    metrics = {}
    # metrics['CTR'] = float(np.sum(ind == 0)) / len(ind) * 100
    metrics['CTR'] = num / count * 100
    return metrics['CTR']