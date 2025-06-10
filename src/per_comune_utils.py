import os
import shutil
import geopandas as gpd
import glob
import zipfile
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gsp
from unidecode import unidecode
from tqdm import tqdm

def loader(a, b, c, d, e):
    istruzione = gpd.read_file(f'/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson/per_comune/ISTRUZIONE/{a}.geojson')
    occupazione = gpd.read_file(f'/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson/per_comune/OCCUPAZIONE/{b}.geojson')
    ifc= gpd.read_file(f'/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson/per_comune/IFC/{c}.geojson')
    popolazione = gpd.read_file(f'/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson/per_comune/popolazione/{d}.geojson')
    redditi = gpd.read_file(f'/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson/per_comune/redditi/{e}.geojson')

    return istruzione, occupazione, ifc, popolazione, redditi


def find_neighbors(gdf, row):
    neighbors = []
    for idx, row2 in gdf.iterrows():
        if (row2['geometry'].touches(row['geometry'])):
            neighbors.append(idx)
    return neighbors


def fill_missing_value(gdf, row, var):
    if pd.isna(row[var]):  # If it is missing...
        neighbors = find_neighbors(gdf, row)
        if neighbors:
            neighbor_val = gdf.loc[neighbors, var]
            # Calculate the average of neighbors (ignoring NaN values)
            if str(gdf[var].dtype) in ['float64']:
                return neighbor_val.mean()
            else:
                m = neighbor_val.mode()
                if m.shape[0] >= 1:
                    return m[0]
    return row[var]

def fix_series_values(row, var):
    if type(row[var]) == pd.core.series.Series:
        return pd.Series(row[var]).iloc[0][0]
    else:
        return row[var]
    
def yearly_data_processing(year, a, b, c, d, e):
    i18, o18, ifc18, p16, r16 = loader(a, b, c, d, e)
    tot16 = i18.merge(o18, how='outer', on='geometry'
                  ).merge(ifc18, how='outer', on='geometry'
                          ).merge(p16, how='outer', on='geometry'
                                  ).merge(r16, how='outer', on='geometry')
    
    tot16.drop(columns=tot16.columns[tot16.isna().all()], inplace=True)
    
    for col in tqdm(tot16.columns, desc='Fixing series values...'):
        tot16.loc[:,col] = tot16[[col]].apply(lambda x: fix_series_values(x, col), axis=1)

    track = 0
    while(tot16.isna().sum().any() & track < 15):
        for col in tqdm(tot16.columns[tot16.isna().any()], desc=f'Fixing missing values time {track}...'):
            tot16.loc[:,col] = tot16[[col, 'geometry']].apply(lambda x: fill_missing_value(tot16, x, col), axis=1)
        track += 1
    
    save_clean_data(tot16, f'per_comune_{year}_provvisorio', '/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson/per_comune')

