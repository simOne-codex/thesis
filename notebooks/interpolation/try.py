import sys
sys.path.append(rf'/nfs/home/genovese/thesis-wildfire-genovese/src/')
from importlib import reload
import utils
reload(utils)
from utils import *
data_folder = '/nfs/home/genovese/thesis-wildfire-genovese/data/'

meteo = dict()
    
for a in list(range(2000, 2025)):
    meteo[a] = gpd.read_file(f'/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson/weather_forecast/{a}.geojson')

meteo_totale = pd.concat(list(meteo.values()))

def year_month_day_to_day_number(year, month, day):
    feb = 28
    if year in list(range(2000, 2025, 4)):
        feb += 1
            
    days_per_month = [31, feb, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_of_year = sum(days_per_month[:month - 1]) + day + year
    return day_of_year
    
sector_to_degree = {
            'Calma di vento': 0,
            'N': 1,
            'NNE': 2,
            'NE': 3,
            'ENE': 4,
            'E': 5,
            'ESE': 6,
            'SE': 7,
            'SSE': 8,
            'S': 9,
            'SSW': 10,
            'SW': 11,
            'WSW': 12,
            'W': 13,
            'WNW': 14,
            'NW': 15,
            'NNW': 16
        }


def DataLoader(gdf, vars = ['tmedia', 'tmax', 'tmin', 'ptot', 'vmedia', 'vraffica', 'settore_prevalente', 'tempo_permanenza',
           'durata_calma', 'umedia', 'umin', 'umax', 'rtot', 'hdd_base18', 'hdd_base20', 'cdd_base18']):
        result = dict.fromkeys(vars)
       
        aux = gdf
        aux['settore_prevalente'] = aux.loc[:, 'settore_prevalente'].map(sector_to_degree)
        aux['day'] = aux.apply(lambda x: year_month_day_to_day_number(x['YYYY'], x['MM'], x['DD']), axis=1)
     
        for key in result.keys():
            foo = aux[['day', 'Quota (m s.l.m.)', key, 'geometry']]
            mask = foo[key].dropna().index 
            foo = foo.loc[mask, :]
     
            result[key] = foo.rename(columns={'Quota (m s.l.m.)': 'height'})
     
        return result

meteo_per_kriging = DataLoader(meteo_totale)


import pickle
with open('data_for_weather_kriging.pkl', 'wb') as f:
     pickle.dump(meteo_per_kriging, f)