# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:35:29 2023

@author: Thomas Pingel
"""

import buildtools as bt
import os

#%%

fn = 'object_list_1674494142583.json'

df = bt.read_blickfeld_log(fn,geometry='line')
df.plot()

lat,lon,rotation = 37.227546,-80.41708,-31
crs = bt.make_bcrs(lat,lon,rotation,'cid_bcrs',method='aeqd')

df.to_file('out/' + fn + '.shp',crs=crs)

