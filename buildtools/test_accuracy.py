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


#%%  CASE 1: 10 random (and randomly paired) points

xy,ab = np.random.rand(10,2), np.random.rand(10,2)
bt.plot_pairs(xy,ab)

bdr_result = bt.bdr(xy,ab)
print('DI: ',bdr_result['DI'])

chamfer_xy_to_ab = bt.chamfer_distance(xy,ab,direction='x_to_y')
chamfer_ab_to_xy = bt.chamfer_distance(xy,ab,direction='y_to_x')
print('chamfer xy_to_ab',chamfer_xy_to_ab,', chamfer ab_to_xy',chamfer_ab_to_xy)
print('chamfer ratio:',chamfer_xy_to_ab,chamfer_ab_to_xy)
print('chamfer ratio normalized:',bt.battersby_normalize(chamfer_xy_to_ab,chamfer_ab_to_xy))




#%% 

row,col,costs = bt.hungarian_algorithm(xy,ab)
ab_paired = ab[col]

bt.plot_pairs(xy,ab_paired)



d = distance(xy,ab_paired)

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

def distance(xy,ab):
    return np.sqrt((xy[:,0]-ab[:,0])**2 + (xy[:,1]-ab[:,1])**2)
    