import pickle
import os 


sample_dir = '/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/cache/samples/'

with open('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/cache/listdir_lists/listdir_samples.pkl', 'wb') as f:
    pickle.load(os.listdir(sample_dir), f)