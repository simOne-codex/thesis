import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score



random_states = [92656, 818, 19963]
cv_rs=1633

## GRIDSEARCH
X_train = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/only_human_study/database/sample_b/pca_X_train_val.csv')
y_train = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/only_human_study/database/sample_b/y_train_val.csv')

# transform = {0.0: 0,
#  0.1: 1,
#  0.2: 2,
#  0.3: 3,
#  0.4: 4,
#  0.5: 5,
#  0.6: 6,
#  0.7: 7,
#  0.8: 8,
#  0.9: 9,
#  1.0: 10}

# y_train['target_encoded'] = y_train.target.round(1).map(transform)

regression_forest = RandomForestRegressor(n_jobs=-1)
param_grid={'max_depth':[int(np.sqrt(X_train.shape[1])), 50, 100], 'min_samples_leaf': [1, 5, 10], 'random_state': random_states}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=cv_rs)
# cv=10

gs_r = GridSearchCV(regression_forest,
                  param_grid=param_grid,
                  cv=cv, 
                  verbose=5, 
                  n_jobs=-1, 
                  scoring='r2')

gs_r.fit(X_train, y_train['target'])

pd.DataFrame(gs_r.cv_results_).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/only_human_study/outputs/sample_b/pca_model_fine_tuning.csv', index=False)
####


########################## CLASSIFIER 

#### GRIDSEARCH

regression_forest = RandomForestClassifier(n_jobs=-1)
param_grid={'max_depth':[int(np.sqrt(X_train.shape[1])), 50, 100], 'min_samples_leaf': [1, 5, 10]}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=cv_rs)
# cv=10

gs_r = GridSearchCV(regression_forest,
                  param_grid=param_grid,
                  cv=cv, 
                  verbose=5, 
                  n_jobs=-1, 
                  scoring='accuracy')

gs_r.fit(X_train, y_train['target'])

pd.DataFrame(gs_r.cv_results_).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/only_human_study/outputs/sample_b/pca_model_classification_fine_tuning.csv', index=False)
####