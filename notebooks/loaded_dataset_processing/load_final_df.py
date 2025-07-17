import sys
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src/OnFireForest")
import dataset
from importlib import reload
reload(dataset)
from dataset import PiedmontDataset, TabularImageDataset

import pickle
import pandas as pd
import copy

# df = PiedmontDataset(root_dir='/nfs/home/genovese/thesis-wildfire-genovese/database/piedmont', compute_stats=True ,apply_augmentations=False)
# with open('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/piedmontdataset_initialised.pkl', 'wb') as f:
#     pickle.dump(df, f)

with open('/nfs/home/genovese/thesis-wildfire-genovese/database/cache/piedmontdataset_initialised.pkl', 'rb') as f:
    df = pickle.load(f)

df2 = copy.deepcopy(df)

nonr = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/final_pca_selected_dataset_non_rekriged.csv')
r = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/final_pca_selected_dataset_rekriged.csv')
ttb_nonr = TabularImageDataset(nonr, df, id_col = 'fire_id')
ttb_r = TabularImageDataset(r, df2, id_col = 'fire_id')



ttb_r.concat(is_fire_idx=True)
ttb_r.df.to_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/final_dataset_rekriged.csv')
ttb_nonr.concat(is_fire_idx=True)
ttb_nonr.df.to_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/final_dataset_non_rekriged.csv')


