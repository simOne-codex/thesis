import sys
sys.path.append(rf'/nfs/home/genovese/thesis-wildfire-genovese/src/')
from importlib import reload
import utils
reload(utils)
from utils import *
data_folder = '/nfs/home/genovese/thesis-wildfire-genovese/data/'

import pickle 
with open('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/data_for_weather_kriging.pkl', 'rb') as f:
     loaded_dict = pickle.load(f)
     
print(list(loaded_dict.keys()))

ignitions = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/kriging_weatehr_grid.geojson'
                       ).rename(columns={'day_of_year': 'day'}) 


from sklearn.ensemble import GradientBoostingRegressor
from pykrige.rk import RegressionKriging

def rk_train_pred(X_train, X_pred, target_col):
    RK = RegressionKriging(regression_model = GradientBoostingRegressor(), method='universal', variogram_model = 'spherical', verbose=1)
       
    RK.fit(p = X_train[['height']], x = np.transpose(np.array([X_train.geometry.x, X_train.geometry.y])), y = X_train[target_col])
       
    y_pred = RK.predict(p = X_pred[['height']], x = np.transpose(np.array([X_pred.geometry.x, X_pred.geometry.y])))

    return y_pred



def yearly_forecast(loaded_dict, target, year):
    
    centraline = loaded_dict[year]
    tt = target[target['YYYY'] == year]

    forecasts = tt.copy()
    
    for var, gdf in tqdm(centraline.items(), desc=f'Running year: {year}'):

        parameter_forecast = gpd.GeoDataFrame()

        for day, X_pred in tqdm(tt.groupby('day'), desc=f'Running variable: {var}'):
            aux = X_pred.copy()
            X_train = gdf[gdf.day == day]
            y_pred = rk_train_pred(X_train, X_pred, var)
            aux = pd.concat([aux, y_pred], axis=1)
            parameter_forecast = pd.concat([parameter_forecast, aux], axis=0, ignore_index=True)

        forecasts.merge(parameter_forecast, on='geometry', how='inner')

    return forecasts



weather_forecast = dict()

for year in tqdm(ignitions.YYYY.unique()):
    weather_forecast[year] = yearly_forecast(loaded_dict, ignitions, year)