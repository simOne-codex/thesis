import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src")
from importlib import reload
import utils
reload(utils)
from utils import *

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
    


gradi_fragilita = {'Minima fragilità': 0,
                   'Molto bassa': 1,
                   'Bassa': 2,
                   'Medio bassa': 3,
                   'Medio-bassa': 3,
                   'Lieve': 4,
                   'Moderata': 5,
                   'Medio alta': 6,
                   'Alta': 7,
                   'Molto alta': 8,
                   'Massima fragilità': 9}
fasce_demografiche = {'fino a 1000 abitanti': 0,
                      '1.000-5.000': 1,
                      '5.001-10.000': 2,
                      '10.001-50.000': 3,
                      '50.001-100.000': 4,
                      '100.001-250.000': 5,
                      'oltre 250.000 abitanti': 6}
grado_u = {'Zone rurali': 0,
           'Piccole città e sobborghi': 1,
           'Città': 2}


cache = '/nfs/home/genovese/thesis-wildfire-genovese/database/cache/'



def yearly_data_processing(year, a, b, c, d, e):
    agricensus = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson/per_comune/agricensus/2020.geojson')
    i18, o18, ifc18, p16, r16 = loader(a, b, c, d, e)
    gdf = i18.merge(o18, how='outer', on='geometry'
                  ).merge(ifc18, how='outer', on='geometry'
                          ).merge(p16, how='outer', on='geometry'
                                  ).merge(r16, how='outer', on='geometry'
                                          ).merge(agricensus, how='outer', on='geometry')
    
    gdf.drop(columns=gdf.columns[gdf.isna().all()], inplace=True)
    
    for col in tqdm(gdf.columns, desc='Fixing series values...'):
        gdf.loc[:,col] = gdf[[col]].apply(lambda x: fix_series_values(x, col), axis=1)
    
    if f'preprocessing_{year}.geojson' not in os.listdir(cache):
        save_clean_data(gpd.GeoDataFrame(gdf), f'preprocessing_{year}', cache)

    track = 0
    while(gdf.isna().sum().any() and (track < 15)):
        if f'preprocessing_{year}.geojson' in os.listdir(cache):
            gdf = gpd.read_file(cache + f'preprocessing_{year}.geojson')
        for col in tqdm(gdf.columns[gdf.isna().any()], desc=f'Fixing missing values time {track}...'):
            gdf.loc[:,col] = gdf[[col, 'geometry']].apply(lambda x: fill_missing_value(gdf, x, col), axis=1)
        track += 1
        gpd.GeoDataFrame(gdf).to_file(cache + f'preprocessing_{year}.geojson', index=False, driver ='GeoJSON')
    

    gdf.rename(columns={'IFC (decil': 'IFC (decile)'}, inplace=True)

    descrizione = [col for col in gdf.columns if 'descriz' in col.lower()][0]
    etampdem = [col for col in gdf.columns if 'et amp' in col.lower()][0]
    etgradou = [col for col in gdf.columns if 'et grado' in col.lower()][0]

    gdf['IFC (valore)'] = gdf[descrizione].map(gradi_fragilita)
    gdf['Fascia demografica'] = gdf[etampdem].map(fasce_demografiche)
    gdf['Grado di urbanizzazione'] = gdf[etgradou].map(grado_u)
    gdf.drop(columns=[descrizione, etampdem, etgradou], inplace=True)


    for col in gdf.columns:
        if str(gdf[col].dtype) == 'object':
            for j in gdf[col].index:
                c = gdf.loc[j, col]
                if isinstance(c, str):
                    if c.isnumeric():
                        gdf.loc[j, col] = float(c.replace(',', '.'))
        
        if col not in ['geometry']:
            if not gdf[col].apply(lambda x: isinstance(x, str)).any():
                gdf[col] = gdf[col].astype('float64')
        
        if '_x' in col:
            gdf.rename(columns={col: col.replace('_x', '_istruzione')}, inplace=True)
        if '_y' in col:
            gdf.rename(columns={col: col.replace('_y', '_occupazione')}, inplace=True)
        

    save_clean_data(gdf, f'per_comune_{year}_provvisorio', '/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson/per_comune')

