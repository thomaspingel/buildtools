# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:13:26 2023

@author: Thomas Pingel
"""

import pandas as pd
import numpy as np
from shapely import geometry
import geopandas
import datetime

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
           b = datetime.datetime.strptime(last_time[:-4],"%Y-%m-%dT%H:%M:%S.%f")
           a = datetime.datetime.strptime(first_time[:-4],"%Y-%m-%dT%H:%M:%S.%f")
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

# TODO: Merge more than one log together