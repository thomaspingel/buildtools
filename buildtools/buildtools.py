# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:13:26 2023

@author: Thomas Pingel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry
import geopandas
from datetime import datetime
import os
from scipy.io import wavfile
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from scipy import stats

import rasterio

#%%

def points_to_lines(df):
    # This isn't a great way to do this, so fix it if you get a chance.
    new_df = df.loc[:,['uuid','timestamp','pose.position.x','pose.position.y','linearVelocity.x','linearVelocity.y']]
    new_df = new_df.rename(columns={'pose.position.x':'x','pose.position.y':'y'})
    new_df['linearVelocity'] = (new_df['linearVelocity.x']**2 + new_df['linearVelocity.y']**2)**.5  
    new_df = new_df.drop(columns=['linearVelocity.x','linearVelocity.y'])
    new_df = new_df.sort_values(by=['uuid','timestamp']) 
    
    lines = []
    
    grp = new_df.groupby('uuid')
    
    names = []
    geometries = []
    velocities = []
    max_velocities = []
    seconds = []
    
    for item in grp:
       name, data = item
       coords = data.loc[:,['x','y']]
       try: 
           g = geometry.LineString(zip(coords.x,coords.y))
    
       
           names.append(name)
           geometries.append(g)
           velocities.append(np.nanmean(data.linearVelocity))
           max_velocities.append(np.nanmax(data.linearVelocity))
           
           last_time = data.iloc[-1]['timestamp']
           first_time = data.iloc[0]['timestamp']
           b = datetime.strptime(last_time[:-4],"%Y-%m-%dT%H:%M:%S.%f")
           a = datetime.strptime(first_time[:-4],"%Y-%m-%dT%H:%M:%S.%f")
           c = (b-a).total_seconds()
           seconds.append(c)    
       except:
           print('fail')           
    lengths = [g.length for g in geometries]
    gdf = geopandas.GeoDataFrame(data={'name':names,'length_m':lengths,'velocity':velocities,'max_velocity':max_velocities,'seconds':seconds},geometry=geometries)

    return gdf


#%%

def read_blickfeld_log(fn,geometry='point'):

    df = pd.read_json(fn)
    df = pd.json_normalize(df['objects'].explode()).drop(columns=['path'])
    
    if geometry=='line':
        df = points_to_lines(df)
    else:
        df = geopandas.GeoDataFrame(df,geometry=geopandas.points_from_xy(df['pose.position.x'], df['pose.position.y']))

    return df


#%%


# Positive azimuths rotate clockwise

def make_bcrs(lat,lon,azimuth,name='local',method='aeqd'):
    if method=='aeqd':
        # This isn't a great way to do this, so fix it when you get a chance.
        a = 'PROJCS["' # + name
        b = '",GEOGCS["GCS_WGS_1984",DATUM["D_WGS84",SPHEROID["WGS84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Local"],PARAMETER["longitude_of_center",' # + lon
        c = '],PARAMETER["latitude_of_center",' # + lat
        d = '],PARAMETER["false_easting",0],PARAMETER["false_northing",0],PARAMETER["azimuth",' # + azimuth
        e = '],UNIT["Meter",1]]'
        return a + name + b + str(lon) + c + str(lat) + d + str(azimuth) + e
    elif method=='omerc':
        return "+proj=omerc +lat_0=" + str(lat) + ' +lonc=' + str(lon) + ' +gamma=' + str(-azimuth) 
    else:
        error('method must be aeqd or omerc')
        
    
    
#%%

def percept2datetime(fn):
    basename = os.path.basename(fn)
    datetime_format = '%Y%m%d-%H%M%S'
    milliseconds = int(basename[12:-5])/1000
    dt = datetime.fromtimestamp(milliseconds)
    return dt, datetime.strftime(dt,datetime_format)


#%%

#%% Bidimensinoal Regression

# Bidimensional Regression: A Python Implementation by Thomas J. Pingel
#
# This Python function is a coding of Friedman and Kohler's 
# (2003) Euclidean bidimensional regression solution.  It was based on an 
# earlier Matlab implementation I did, now on GitHub at 
# https://github.com/thomaspingel/bidimensional-regression-matlab
#
# References: 
# Friedman, A., & Kohler, B. (2003). Bidimensional Regression: Assessing
#   the Configural Similarity and Accuracy of Cognitive Maps and Other Two-Dimensional 
#   Data Sets. Psychological Methods, 8(4), 468-491.
#   https://doi.org/10.1037/1082-989X.8.4.468
#   http://www.psych.ualberta.ca/~alinda/PDFs/Friedman%20Kohler%20%5B03-Psych%20Methods%5D.pdf
#
# Nakaya, T. (1997). Statistical inferences in bidimensional regression
#   models. Geographical Analysis, 29, 169-186.
#   https://doi.org/10.1111/j.1538-4632.1997.tb00954.x
#   
#
# Tobler, W. (1994). Bidimensional Regression. Geographical Analysis, 26, 187-212.
#   https://doi.org/10.1111/j.1538-4632.1994.tb00320.x 
#
# This code copyrighted June 14, 2023.
#
# Free for use, but please cite if used for data prepared for publication.
# Thomas. J Pingel
# Department of Geography, Binghamton University
# tpingel@binghamton.edu
# The output is a structure containing the parameters of the regression and distortion.  These are:
#
# n -           Number of points used in the analysis 
# beta1 -       Regression parameter
# beta2 -       Regression parameter
# alpha1 -      Translation dispacement for X
# alpha2 -      Translation displacement for Y
# scale (phi) - A value greater than one means dependent is larger (expansion), 
#               a value smaller than one indicates dependent coordinates are 
#               smaller (contraction).
# theta -       A measure of angular displacement.  Negative indicates a
#               clockwise rotation is necessary.
# aPrime -      Predicted X coordinates
# bPrime -      Predicted Y coordinates
# rsquare -     Overall amount of variance explained by regression
# D -           Square root of the unexplained variance of between
#               dependent and predicted values (AB and ABprime)
# Dmax -        Maximum value of D.
# DI -          Distortion Index.  A measure of unexplained variance.
# F -           F-ratio, from Nakaya, 1997, Equation 50
# P -           p value for F, also from Nakaya.
# 
# Of these, alphas, phi, and theta are likely of most use to the researcher
# as specific descriptions of parts of the distortion.
# Rsquare and DI are likely the most useful as overall or general measures of
# distortion.

def bdr(XY,AB):
    
    X = XY[:,0]
    Y = XY[:,1]
    A = AB[:,0]
    B = AB[:,1]
    
    def ssq(x):
        return np.sum((x-np.mean(x))**2);

    # These equations are given in Table 2, page 475 of Friedman and Kohler
    beta1 = (np.sum((X-np.mean(X))*(A-np.mean(A))) + np.sum((Y-np.mean(Y))*(B-np.mean(B))))/(ssq(X)+ssq(Y))
    beta2 = (np.sum((X-np.mean(X))*(B-np.mean(B))) - np.sum((Y-np.mean(Y))*(A-np.mean(A))))/(ssq(X)+ssq(Y))
    scale = (beta1**2 + beta2**2)**.5
    theta = np.rad2deg(np.arctan2(beta2,beta1))
    alpha1 = np.mean(A) - beta1*np.mean(X) + beta2*np.mean(Y)
    alpha2 = np.mean(B) - beta2*np.mean(X) - beta1*np.mean(Y)
    aPrime = np.array(alpha1 + beta1*X - beta2*Y)   # np.arrays because might come in as a dataframe
    bPrime = np.array(alpha2 + beta2*X + beta1*Y)
    rsquare = 1 - np.sum((A-aPrime)**2 + (B-bPrime)**2)/np.sum(ssq(A)+ssq(B))
    D = np.sqrt(np.sum((A-aPrime)**2 + (B-bPrime)**2))  # Equivalent to D_AB in the article
    Dmax = np.sqrt(ssq(A) + ssq(B))                     # Equivalent to Dmax_AB
    # Dmax = sqrt(ssq(X) + ssq(Y));                # Note that these are square roots of equivalent numbers in table 3 of paper
    DI = np.sqrt(1-rsquare)                        # This is how DI is computed in sample and equation 12, page 479
    F = ((2*len(A) - 4) / (4 - 2)) * (rsquare / (1-rsquare))
    P = 1 - stats.f.cdf(F,2,2*len(A)-4)
    
    result = {}
    result['beta1']=beta1; result['beta2']=beta2; result['alpha1']=alpha1; 
    result['alpha2']=alpha2; result['scale']=scale; result['theta']=theta;
    result['aPrime']=aPrime; result['bPrime']=bPrime; result['rsquare']=rsquare; 
    result['D']=D; result['Dmax']=Dmax; result['DI']=DI; result['F']=F; result['P']=P
    
    return result

#%%  From: https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367

def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist


#%%


def hungarian_algorithm(XY,AB):
    cost_matrix = cdist(XY,AB)
    # Perform the Hungarian algorithm using linear_sum_assignment function from scipy.optimize
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    min_costs = cost_matrix[row_indices, col_indices]
    
    # Return the optimal assignment and the minimum cost
    return row_indices, col_indices, min_costs

#%%

def bdr_bootstrap(XY,AB,k=10000):
   rsquare = np.zeros(k)
   DI = np.zeros(k)
   for i in range(k):
       idx = np.random.choice(len(AB),len(XY),replace=False)
       ABs = AB[idx,:]
       row,col,costs = hungarian_algorithm(XY,ABs)
       bdr_result = bdr(XY,ABs[col,:])
       rsquare[i] = bdr_result['rsquare']
       DI[i] = bdr_result['DI']
   return rsquare, DI   

def gnomonic_projection(lon, lat, lon0, lat0, trim = True):
 """
 Performs a gnomonic projection on a set of longitude and latitude points.
 Parameters:
 lon (numpy.ndarray): Array of longitude values in degrees.
 lat (numpy.ndarray): Array of latitude values in degrees.
 lon0 (float): Central longitude for the projection in degrees.
 lat0 (float): Central latitude for the projection in degrees.
 radius (float): height
 Returns:
 x (numpy.ndarray): Array of x values in projected space.
 y (numpy.ndarray): Array of y values in projected space.
 """
 angle = angle_between_points(lon,lat,lon0,lat0)

 # Convert longitude and latitude to radians
 lon = np.radians(lon)
 lat = np.radians(lat)
 lon0 = np.radians(lon0)
 lat0 = np.radians(lat0)
 # Calculate the distance from the central point
 cos_c = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)
 k = 1 / cos_c
 # Calculate the x and y coordinates in the projected space
 x = k * np.cos(lat) * np.sin(lon - lon0)
 y = k * (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0))
 
 if trim:
     x = x[angle<=90]
     y = y[angle<=90]
 else:
     x[angle>=90] = np.nan
     y[angle>=90] = np.nan
 
 return x, y


# THIS IS THE GOOD ONE.  CHECKS OUT AGAINST MATLAB ALGORITH FOR SPH2CART
# AND MATCHES THE VIRUTAL MIC WHEN ELEVATION=0
# This is also equivalent to panning a source
# https://en.wikipedia.org/wiki/Ambisonics#Panning_a_source
# https://en.wikipedia.org/wiki/Ambisonics#Virtual_microphones

def virtual_mic(x, y, z, w, azimuth, elevation, p=.5):
    
    # Convert azimuth and elevation to radians
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)
    
    if not np.isscalar(azimuth):
        azimuth = azimuth[:,:,np.newaxis]
        elevation = elevation[:,:,np.newaxis]

    # Convert to Cartesian coordinates
    x_cart = x * np.cos(elevation) * np.cos(azimuth)
    y_cart = y * np.cos(elevation) * np.sin(azimuth)
    z_cart = z * np.sin(elevation)

    # Calculate intensity as magnitude of the Cartesian coordinates
    intensity = x_cart + y_cart + z_cart

    # Virtual mic has the w side multiplied by square root of two
    intensity = p*(w * (2**.5)) + (1-p)*intensity

    return intensity.astype(x.dtype)

#%%
# a,b,c are x,y,z respectively.
def virtual_point(x,y,z,w,XI,YI,ZI,p=.5):
    if not np.isscalar(XI):
        XI = XI[:,:,np.newaxis]
    if not np.isscalar(YI):
        YI = YI[:,:,np.newaxis]
    if not np.isscalar(ZI):
        ZI = ZI[:,:,np.newaxis]
    d = np.sqrt(XI**2 + YI**2 + ZI**2)
    intensity = (x * XI / d)  +  (y * YI / d)  +  (z * ZI / d)
    intensity = p*(w * (2**.5)) + (1-p)*intensity
    
    return intensity.astype(x.dtype)
    

#%%

def read_wav(fn):
 rate,x = wavfile.read(fn)
 x=x[1:,:]  # Not the first row?  This doesn't seem to be data.
 w,x,y,z = x.T
 t = np.arange(len(w)) / rate
 return rate, pd.DataFrame({'x':x,'y':y,'z':z,'w':w,'t':t})

#%%


# The transform:
# t = rasterio.transform.from_origin(-180,90,resolution,resolution)

def equirectangular(df,p=.5,resolution=1,i=None,t=None,radius=None):
    if i is None and t is None:
        print('i or t must be provided')
    if i is not None and t is not None:
        print('i AND t cannot both be provided.')
    if t is not None:
        if t > df['t'].iloc[-1]:
            print('t must not be greater than the last value in df')
        else:
            i = np.argmin(np.abs(df.t - t))

    lon = np.arange(-180,180,resolution) + resolution/2
    lat = np.arange(90,-90,-resolution)
    LON,LAT = np.meshgrid(lon,lat) 
    
    # x,y,z,w
    if radius is None:
        field = virtual_mic(*df.iloc[i].values[:-1],LON,LAT,p)
    else:
        a = i - radius
        b = i + radius + 1
        if a < 0: a = 0
        if b>=len(df): b = len(df)-1
        x,y,z,w = df.iloc[a:b,0:4].values.T
        field = virtual_mic(x,y,z,w,LON,LAT,p)
        field = np.mean(np.abs(field),axis=2)
        
    field = np.squeeze(field)
            
    return field

#%%

def project(df,width=1,height=1,resolution=.1,p=.5,i=None,t=None,radius=0,method='cartesian'):
    if i is None and t is None:
        print('i or t must be provided')
    if i is not None and t is not None:
        print('i AND t cannot both be provided.')
    if t is not None:
        if t > df['t'].iloc[-1]:
            print('t must not be greater than the last value in df')
        else:
            i = np.argmin(np.abs(df.t - t))
            
    print(i)

    if radius is None:
        if method=='cartesian':
            xi = np.arange(-width,width+resolution,resolution)
            yi = np.arange(width,-width-resolution,-resolution)
            XI,YI = np.meshgrid(xi,yi)
            ZI = height * np.ones_like(XI)
            field = virtual_point(*df.iloc[i].values[:-1],XI,YI,ZI,p)
        elif method=='spherical':
            lon = np.arange(-180,180,resolution)
            lat = np.arange(90,-90,-resolution)
            LON,LAT = np.meshgrid(lon,lat) 
            field = virtual_mic(*df.iloc[i].values[:-1],LON,LAT,p)
        else:
            print('must specify cartesian or spherical for method')
    else:
        a = i - radius
        b = i + radius + 1
        if a < 0: a = 0
        if b>=len(df): b = len(df)-1
        x,y,z,w = df.iloc[a:b,0:4].values.T
        if method=='cartesian':
            xi = np.arange(-width,width+resolution,resolution)
            yi = np.arange(width,-width-resolution,-resolution)
            XI,YI = np.meshgrid(xi,yi)
            ZI = height * np.ones_like(XI)
            field = virtual_point(x,y,z,w,XI,YI,ZI,p)
        elif method=='spherical':
            lon = np.arange(-180,180,resolution)
            lat = np.arange(90,-90,-resolution)
            LON,LAT = np.meshgrid(lon,lat) 
            field = virtual_mic(x,y,z,w,LON,LAT,p)
        else:
            print('must specify cartesian or spherical for method')
        field = np.mean(np.abs(field),axis=2)
        
    return field


def score(A,B,k=100000,mask=None):
    if mask is None:
        A = A.flatten()
        B = B.flatten()
    else:
        A = A[mask].flatten()
        B = B[mask].flatten()
        
    if k > len(A):
        k = len(A)
        
    s = np.random.choice(len(A),k,replace=True)
    kappa = cohen_kappa_score(A[s],B[s])
    cmatrix = confusion_matrix(A[s],B[s])
    f1 = f1_score(A[s],B[s])
    ac = accuracy_score(A[s],B[s])
    
    result = {'cohen_kappa_score':kappa,
              'confusion_matrix':cmatrix,
              'f1_score':f1,
              'accuracy_score':ac}
    
    return result

#%%

def plot_pairs(a,b,hungarian_indexes=None):
    plt.plot(*a.T,'ko')
    plt.plot(*b.T,'rx')
    if hungarian_indexes is not None:
        if len(a)<=len(b):
            b = b[hungarian_indexes]
        else:
            a = a[hungarian_indexes]
    for point_a, point_b in zip(a, b):
        plt.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], color='black')
        
#%%

# See Battersby. 2009. The Effect of Global-Scale Map-Projection Knowledge on Perceived Land Area. doi.org/10.3138/carto.44.1.33
# Page 38, equations 1 and 2

def battersby_normalize(a,b):
    if a>b:
        return a/b - 1
    else:
        return -(b/a - 1)
    
    
#%%


def build_score(xy,ab,t):
    
    results = {}
    results['threshold'] = t
    results['n_xy'] = len(xy)
    results['b_ab'] = len(ab)
    results['xy'] = xy
    results['ab'] = ab
    
    # Chamfer
    chamfer_xy_to_ab = chamfer_distance(xy,ab,direction='x_to_y')
    chamfer_ab_to_xy = chamfer_distance(xy,ab,direction='y_to_x')
    if chamfer_xy_to_ab >= chamfer_ab_to_xy:
        chamfer_ratio = chamfer_xy_to_ab / chamfer_ab_to_xy - 1
    else:
        chamfer_ratio = -(chamfer_ab_to_xy / chamfer_xy_to_ab - 1)
    results['chamfer_xy_to_ab'] = chamfer_xy_to_ab
    results['chamfer_ab_to_xy'] = chamfer_ab_to_xy
    results['chamfer_ratio'] =  chamfer_ratio_xy/chamfer_ratio_ab
    results['chamfer_ratio_battersby']    = chamfer_ratio    
    results['chamfer_ratio_log']  = np.log(chamfer_ratio_xy/chamfer_ratio_ab)    

    if len(xy) <= len(ab):  # At least as many object detected as exist
        row,col,costs = hungarian_algorithm(xy,ab)
        ab_paired = ab[col]
        bdr_result = bdr(xy,ab_paired)
        d = np.linalg.norm(xy - ab_paired, axis=1)         # Distance
    else:                   # Fewer objects detected than exist
        row,col,costs = hungarian_algorithm(ab,xy)
        xy_paired = xy[col]
        bdr_result = bdr(xy_paired,ab)
        d = np.linalg.norm(xy_paired - ab, axis=1) 
        
    results['bdr'] = bdr_result
    results['distance'] = d
    results['hungarian_indexes'] = col            # This is the Hungarian Distance lookup

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
    

def pix2coords(cols,rows):
    gt = rasterio.transform.from_origin(-12., 25., .05, .05)
    map_x,map_y = gt * (cols,rows)
    # Alternatively: map_x, map_y = rasterio.transform.xy(gt,rows,cols,offset='ul')    
    return map_x, map_y

def coords2pix(x,y):
    gt = rasterio.transform.from_origin(-12., 25., .05, .05)
    cols, rows = ~gt * (x,y)
    return cols, rows

def hello(s):
    return "hi " + s
    
