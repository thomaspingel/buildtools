# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 22:57:04 2023

@author: thoma
"""

import buildtools as bt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from scipy.spatial.distance import pdist

#%%

def distance(xy,ab):
    return np.sqrt((xy[:,0]-ab[:,0])**2 + (xy[:,1]-ab[:,1])**2)

    # or: np.linalg.norm(xy - ab, axis=1)
    # or: np.sqrt(np.sum((xy-ab)**2,axis=1))


#%%  CASE 1: 10 random (and randomly paired) points

n_xy = 10
n_ab = 10

xy,ab = np.random.rand(n_xy,2), np.random.rand(n_ab,2)
bt.plot_pairs(xy,ab)

bdr_result = bt.bdr(xy,ab)
print('DI: ',bdr_result['DI'])

chamfer_xy_to_ab = bt.chamfer_distance(xy,ab,direction='x_to_y')
chamfer_ab_to_xy = bt.chamfer_distance(xy,ab,direction='y_to_x')
print('chamfer xy_to_ab',chamfer_xy_to_ab,', chamfer ab_to_xy',chamfer_ab_to_xy)
print('chamfer ratio:',chamfer_xy_to_ab,chamfer_ab_to_xy)
print('chamfer ratio normalized:',bt.battersby_normalize(chamfer_xy_to_ab,chamfer_ab_to_xy))

result = bt.build_score(xy,ab,t=.1)
print(result)



#%% Case 2: 10 random points, paired using Hungarian Algorithm (min cost)

n_xy = 10
n_ab = 10

xy,ab = np.random.rand(n_xy,2), np.random.rand(n_ab,2)
row,col,costs = bt.hungarian_algorithm(xy,ab)
ab = ab[col]

bt.plot_pairs(xy,ab)

bdr_result = bt.bdr(xy,ab)
print('DI: ',bdr_result['DI'])

chamfer_xy_to_ab = bt.chamfer_distance(xy,ab,direction='x_to_y')
chamfer_ab_to_xy = bt.chamfer_distance(xy,ab,direction='y_to_x')
print('chamfer xy_to_ab',chamfer_xy_to_ab,', chamfer ab_to_xy',chamfer_ab_to_xy)
print('chamfer ratio:',chamfer_xy_to_ab,chamfer_ab_to_xy)
print('chamfer ratio normalized:',bt.battersby_normalize(chamfer_xy_to_ab,chamfer_ab_to_xy))


d = np.linalg.norm(xy - ab, axis=1)  # Distance calculation

t = .2

TN = 0
TP = np.sum(d<t)
FN = len(xy) - TP
FP = len(ab) - TP

n = TP + FN + FP

precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * (precision * recall) / (precision + recall)

print('accuracy',np.round(TP / (n),3))
print('precision',precision)
print('recall',recall)
print('F1',np.round(F1,3))    
print('TP',TP)
print('FN',FN)
print('FP',FP)

#%%

# Case 3: More observations in ab

n_xy = 10
n_ab = 100

xy,ab = np.random.rand(n_xy,2), np.random.rand(n_ab,2)
plt.plot(*ab.T,'x')
row,col,costs = bt.hungarian_algorithm(xy,ab)
ab_paired = ab[col]

bt.plot_pairs(xy,ab_paired)

bdr_result = bt.bdr(xy,ab_paired)
print('DI: ',bdr_result['DI'])

chamfer_xy_to_ab = bt.chamfer_distance(xy,ab,direction='x_to_y')
chamfer_ab_to_xy = bt.chamfer_distance(xy,ab,direction='y_to_x')
print('chamfer xy_to_ab',chamfer_xy_to_ab,', chamfer ab_to_xy',chamfer_ab_to_xy)
print('chamfer ratio:',chamfer_xy_to_ab,chamfer_ab_to_xy)
print('chamfer ratio normalized:',bt.battersby_normalize(chamfer_xy_to_ab,chamfer_ab_to_xy))


d = np.linalg.norm(xy - ab_paired, axis=1)  # Distance calculation

t = .05

TN = 0
TP = np.sum(d<t)
FN = len(xy) - TP
FP = len(ab) - TP

n = TP + FN + FP


precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * (precision * recall) / (precision + recall)

#print('accuracy',np.round(TP / len(xy),3))
print('precision',precision)
print('recall',recall)
print('F1',np.round(F1,3))    
print('TP',TP)
print('FN',FN)
print('FP',FP)

#%%

# Case 4: More observations in xy

n_xy = 10
n_ab = 8

xy,ab = np.random.rand(n_xy,2), np.random.rand(n_ab,2)
plt.plot(*xy.T,'.')
plt.plot(*ab.T,'x')
row,col,costs = bt.hungarian_algorithm(ab,xy)
xy_paired = xy[col]

bt.plot_pairs(xy_paired,ab)

bdr_result = bt.bdr(xy_paired,ab)
print('DI: ',bdr_result['DI'])

chamfer_xy_to_ab = bt.chamfer_distance(xy,ab,direction='x_to_y')
chamfer_ab_to_xy = bt.chamfer_distance(xy,ab,direction='y_to_x')
print('chamfer xy_to_ab',chamfer_xy_to_ab,', chamfer ab_to_xy',chamfer_ab_to_xy)
print('chamfer ratio:',chamfer_xy_to_ab/chamfer_ab_to_xy)
print('chamfer ratio normalized:',bt.battersby_normalize(chamfer_xy_to_ab,chamfer_ab_to_xy))

d = np.linalg.norm(xy_paired - ab, axis=1)  # Distance calculation

t = .25

TN = 0
TP = np.sum(d<t)
FN = len(xy) - TP
FP = len(ab) - TP

n = TP + FN + FP

precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * (precision * recall) / (precision + recall)

#print('accuracy',np.round(TP / len(xy),3))
print('precision',precision)
print('recall',recall)
print('F1',np.round(F1,3))    
print('TP',TP)
print('FN',FN)
print('FP',FP)


#%%

def build_score(xy,ab,t):
    
    results = {}
    
    # Chamfer
    chamfer_xy_to_ab = bt.chamfer_distance(xy,ab,direction='x_to_y')
    chamfer_ab_to_xy = bt.chamfer_distance(xy,ab,direction='y_to_x')
    if chamfer_xy_to_ab >= chamfer_ab_to_xy:
        chamfer_ratio = chamfer_xy_to_ab / chamfer_ab_to_xy - 1
    else:
        chamfer_ratio = -(chamfer_ab_to_xy / chamfer_xy_to_ab - 1)
    results['chamfer_xy_to_ab'] = chamfer_xy_to_ab
    results['chamfer_ab_to_xy'] = chamfer_ab_to_xy
    results['chamfer_ratio']    = chamfer_ratio    

    if len(xy) <= len(ab):  # At least as many object detected as exist
        row,col,costs = bt.hungarian_algorithm(xy,ab)
        ab_paired = ab[col]
        bdr_result = bt.bdr(xy,ab_paired)
        d = np.linalg.norm(xy - ab_paired, axis=1)         # Distance
    else:                   # Fewer objects detected than exist
        row,col,costs = bt.hungarian_algorithm(ab,xy)
        xy_paired = xy[col]
        bdr_result = bt.bdr(xy_paired,ab)
        d = np.linalg.norm(xy_paired - ab, axis=1) 
        
    results['bdr'] = bdr_result
    results['distance'] = d
    results['col'] = col            # This is the Hungarian Distance lookup

    TP = np.sum(d<t)
    FN = len(xy) - TP
    FP = len(ab) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN) 
    F1 = 2 * (precision * recall) / (precision + recall)
    results['n'] = TP + FN + FP
    results['TP']=TP;results['FN']=FN;results['FP']=FP;results['precision']=precision
    results['recall']=recall;results['F1']=F1;
    
    return results
    
    

        