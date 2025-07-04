import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src")
from importlib import reload
import utils
reload(utils)
from utils import *

gdf = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/dataset.geojson')

def daylize(year, month, day):
    feb = 28
    if int(year) in list(range(2000, 2025, 4)):
        feb += 1
    
    years = pd.Series([0, 366, 365, 365, 365, 366, 365, 365, 365])
    months = pd.Series([0, 31, feb, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

    year_component = years.loc[:int(year)-2016].sum()
    month_component = months.loc[:int(month)-1].sum()
    
    return year_component+month_component+day

gdf['day_count'] = gdf.apply(lambda row: daylize(row.YYYY, row.MM, row.DD), axis=1)

data = gdf.drop(columns=['geometry', 'DD', 'MM', 'YYYY'])

save_clean_data(data, 'dataset_with_days_enumerated', '/nfs/home/genovese/thesis-wildfire-genovese/database/model_input', force=True)

from ydata_profiling import ProfileReport

profile = ProfileReport(data, title='EDA.html', explorative=True, interactions={"continuous": False}) #disable pairplots

profile.to_file('/nfs/home/genovese/thesis-wildfire-genovese/outputs/EDA_ProfileReport.html')