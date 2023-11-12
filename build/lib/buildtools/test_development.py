# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:12:10 2023

@author: Thomas Pingel
"""

import buildtools as bt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

#%%

dir(bt)

#%%

fn = 'https://raw.githubusercontent.com/thomaspingel/buildtools/main/test/object_list_1688863634095.json'

#%%

as_datetime, as_string = bt.percept2datetime(fn)
print('Filename is:', os.path.basename(fn))
print('Files get tagged as number of microseconds since epoch.')
print(as_datetime)
print(as_string)

#%%

df = bt.read_blickfeld_log(fn)

#%%

df2 = bt.points_to_multipoints(df)
