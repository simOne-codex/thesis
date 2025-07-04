
import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src")
from importlib import reload
import utils
reload(utils)
from utils import *

data = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/cache/processed_not_imputed_non_rekriged.csv').reset_index(drop=True)
data2 = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/cache/processed_not_imputed_non_rekriged.csv').reset_index(drop=True)

target1 = data[['fire_id', 'target']]
target2 = data2[['fire_id', 'target']]

data_auxiliar = data.copy()
data_auxiliar = data_auxiliar.loc[:, [col for col in data.columns if col not in ['target', 'fire_id']]]

from sklearn.impute import KNNImputer
import pickle

imputer = KNNImputer(n_neighbors=10, weights='distance')
imputed_array = imputer.fit_transform(data_auxiliar)

with open('/nfs/home/genovese/thesis-wildfire-genovese/src/trained_models/knn_imputer.pkl', 'wb') as f:
    pickle.dump(imputed_array, f)

df_imputed = pd.DataFrame(imputed_array, columns=data_auxiliar.columns, index=data_auxiliar.index)

df_final1 = df_imputed.assign(fire_id = target1['fire_id'])
df_final2 = df_imputed.assign(fire_id = target2['fire_id'])

df_final1 = df_final1.assign(target = target1['target'])
df_final2 = df_final2.assign(target = target2['target'])

save_clean_data(df_final1, 'imputed_dataset_non_kriged', '/nfs/home/genovese/thesis-wildfire-genovese/database/model_input', force=True)
save_clean_data(df_final2, 'imputed_dataset_kriged', '/nfs/home/genovese/thesis-wildfire-genovese/database/model_input', force=True)
