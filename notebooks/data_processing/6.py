import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src")
from importlib import reload
import utils
reload(utils)
from utils import *
import per_comune_utils
reload(per_comune_utils)
from per_comune_utils import *
data_folder = "/nfs/home/genovese/thesis-wildfire-genovese/data/"

agricensus = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson/per_comune/agricensus/2020.geojson')

yearly_data_processing(2022, 2022, 2022,2021, 2019, 2022)