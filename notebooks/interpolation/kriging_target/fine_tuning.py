import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src")
from importlib import reload
import utils
reload(utils)
from utils import *

positives = gpd.GeoDataFrame(
    pd.DataFrame(gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/kriging_weather_grid.geojson')),
    ).set_crs(epsg=3857, allow_override=True)
negatives = gpd.GeoDataFrame(pd.DataFrame(
    gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/kriging_negative_weather_grid.geojson')),
    crs='EPSG:3857')

p = pd.concat([positives[['geometry']], pd.Series([1]*positives.shape[0], name='label')],  axis=1)
n = pd.concat([negatives[['geometry']], pd.Series([0]*negatives.shape[0], name='label')],  axis=1)

tot = pd.concat([p, n])


import numpy as np
from pykrige.rk import Krige
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Define parameter grid for Kriging
psills = [0.5, 1, 1.5]
ranges = [0.5, 1, 1.5]
nuggets = [-1, 0, 1]
param_grid = {
    "nlags": [4, 6, 8], # Number of averaging bins for the semivariogram. Default is 6.
    "weight": [True, False], # Flag that specifies if semivariance at smaller lags should be weighted more heavily when automatically calculating variogram model. The routine is currently hard-coded such that the weights are calculated from a logistic function
    'variogram_parameters': [{'psill': psills[0], 'range': ranges[0], 'nugget': nuggets[0]},
                             {'psill': psills[0], 'range': ranges[0], 'nugget': nuggets[1]},
                             {'psill': psills[0], 'range': ranges[0], 'nugget': nuggets[2]},
                             {'psill': psills[0], 'range': ranges[1], 'nugget': nuggets[0]},
                             {'psill': psills[0], 'range': ranges[1], 'nugget': nuggets[1]},
                             {'psill': psills[0], 'range': ranges[1], 'nugget': nuggets[2]},
                             {'psill': psills[0], 'range': ranges[2], 'nugget': nuggets[0]},
                             {'psill': psills[0], 'range': ranges[2], 'nugget': nuggets[1]},
                             {'psill': psills[0], 'range': ranges[2], 'nugget': nuggets[2]},
                             {'psill': psills[1], 'range': ranges[0], 'nugget': nuggets[0]},
                             {'psill': psills[1], 'range': ranges[0], 'nugget': nuggets[1]},
                             {'psill': psills[1], 'range': ranges[0], 'nugget': nuggets[2]},
                             {'psill': psills[1], 'range': ranges[1], 'nugget': nuggets[0]},
                             {'psill': psills[1], 'range': ranges[1], 'nugget': nuggets[1]},
                             {'psill': psills[1], 'range': ranges[1], 'nugget': nuggets[2]},
                             {'psill': psills[1], 'range': ranges[2], 'nugget': nuggets[0]},
                             {'psill': psills[1], 'range': ranges[2], 'nugget': nuggets[1]},
                             {'psill': psills[1], 'range': ranges[2], 'nugget': nuggets[2]},
                             {'psill': psills[2], 'range': ranges[0], 'nugget': nuggets[0]},
                             {'psill': psills[2], 'range': ranges[0], 'nugget': nuggets[1]},
                             {'psill': psills[2], 'range': ranges[0], 'nugget': nuggets[2]},
                             {'psill': psills[2], 'range': ranges[1], 'nugget': nuggets[0]},
                             {'psill': psills[2], 'range': ranges[1], 'nugget': nuggets[1]},
                             {'psill': psills[2], 'range': ranges[1], 'nugget': nuggets[2]},
                             {'psill': psills[2], 'range': ranges[2], 'nugget': nuggets[0]},
                             {'psill': psills[2], 'range': ranges[2], 'nugget': nuggets[1]},
                             {'psill': psills[2], 'range': ranges[2], 'nugget': nuggets[2]}]
}

# Create Kriging model (scikit-learn wrapper)
kriging_model = Krige(method='ordinary', variogram_model='spherical')

# GridSearch with 10-fold cross-validation
grid_search = GridSearchCV(
    estimator=kriging_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=10,
    verbose=10,
    n_jobs=-1
)

# Fit GridSearch
grid_search.fit(np.array(pd.concat([tot.geometry.x, tot.geometry.y], axis=1)), tot.label)
result = pd.DataFrame(grid_search.cv_results_)

folder_path = '/nfs/home/genovese/thesis-wildfire-genovese/outputs/grid_searches'
file_name = "fine_tuning_target_kriging"
os.makedirs(folder_path, exist_ok=True)
save_clean_data(result, file_name, folder_path)
