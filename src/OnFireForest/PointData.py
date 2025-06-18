import os
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm


class PointData():

    def __init__(self):
        self.__data = None

    def get_table(self):
        return self.__data

    def __variable_percentage(self, buffer, clipped, var):
        areas_ratios = dict.fromkeys(clipped[var].unique())
        buffer_area = buffer.area[0]

        for value in areas_ratios.keys():
            piece = clipped[clipped[var] == value]
            areas_ratios[value] = (piece.area.sum()/buffer_area)
    
        return areas_ratios


    def __polygon_coverage_in_buffer(self, gdf, query_point, vars, radius_meters):

        gdf = gdf.to_crs(epsg=3857)  # Web Mercator in meters
        query_point = gpd.GeoSeries([query_point], crs="EPSG:3857").iloc[0]
        # Create buffer around the point
        buffer = gpd.GeoDataFrame([query_point.buffer(radius_meters)], columns=['geometry'], crs='EPSG:3857')
        # Intersect polygon(s) with buffer
        clipped = gpd.overlay(gdf, buffer, how='intersection')
        # Remove empty geometries
        clipped = clipped[~clipped.is_empty]
    
        # Calculate percentage coverage
        result = gpd.GeoSeries([query_point], index = ['geometry'], crs="EPSG:3857")

        # Compute ratios for every variable value
        for var in vars:
            areas_ratios = self.__variable_percentage(buffer, clipped, var)
            for key, ratio in areas_ratios.items():
                name = str(var)+'_'+str(key)
                new_line = pd.Series([ratio], index=[name])
                result = pd.concat([result, new_line])

        return pd.Series([result.drop(columns='geometry')])


    def __missing_georeference(self, df):
        return (df.shape[0] > 0)


    def gather_point_data(self, per_comune_gdf, altimetria_gdf, vegetazione_gdf, strade_gdf, meteo_gdf, day, query_point, radius_meters=1000):

        # dati per comune
        aux = per_comune_gdf[per_comune_gdf.contains(query_point)]
        # if self.__missing_georeference(aux):
        #     cols = [col for col in per_comune_gdf.columns if col not in ['geometry']]
        #     data_per_comune = pd.Series([None]*len(cols), index = cols)
        # else:
        data_per_comune = pd.Series(aux.loc[aux.index[0], [col for col in aux.columns if col not in ['geometry']]])
        
        # altimetria
        aux = altimetria_gdf[altimetria_gdf.contains(query_point)]
        # if self.__missing_georeference(aux):
        #     cols = [col for col in altimetria_gdf.columns if col not in ['geometry']]
        #     data_altimetria = pd.Series([None]*len(cols), index = cols)
        # else:
        data_altimetria = pd.Series(aux.loc[aux.index[0], [col for col in aux.columns if col not in ['geometry']]])

        # copertura vegetazione nel raggio di radius_meters
        data_vegetazione = pd.Series(self.__polygon_coverage_in_buffer(vegetazione_gdf, 
                                                             query_point, 
                                                             [col for col in vegetazione_gdf.columns if col not in ['geometry']],
                                                              radius_meters))
        # copertura strade nel raggio di radius_meters
        data_strade = pd.Series(self.__polygon_coverage_in_buffer(strade_gdf, 
                                                        query_point, 
                                                        [col for col in strade_gdf.columns if col not in ['geometry']],
                                                        radius_meters))
        # previsioni meteo del punto
        aux = meteo_gdf.merge(gpd.GeoDataFrame({'geometry':[query_point]}, geometry='geometry'), on='geometry', how='inner')
        # if self.__missing_georeference(aux):
        #     cols = [col for col in meteo_gdf.columns if col not in ['geometry', 'day', 'YYYY', 'height']]
        #     data_meteo = pd.Series([None]*len(cols), index = cols)
        # else:
        data_meteo = pd.Series(aux[aux.day == day].loc[aux[aux.day == day].index[0],[col for col in aux.columns if col not in ['geometry', 'day', 'YYYY', 'height']]])


        data = pd.concat([data_per_comune, data_altimetria, data_vegetazione, data_strade, data_meteo], axis=0)

        self.__data = pd.DataFrame(data.drop(index=[0])).T.astype('float64')