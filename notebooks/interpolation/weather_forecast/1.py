import pickle 
with open('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/data_for_weather_kriging.pkl', 'rb') as f:
     loaded_dict = pickle.load(f)
     
print(list(loaded_dict.keys()))

ignitions = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/kriging_weatehr_grid.geojson'
                       ).rename(columns={'day_of_year': 'day'}) 

save_clean_data(yearly_forecast(loaded_dict, ignitions, 2016), str(2016), '/nfs/home/genovese/thesis-wildfire-genovese/data/output_weather_kriging')