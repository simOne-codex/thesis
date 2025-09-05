import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

random_state = 92656
cv_rs=1633

X_train_val = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/pca_study/database/sample_a/final_X_train_val.csv')
y_train_val = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/pca_study/database/sample_a/y_train_val.csv')

regression_forest = RandomForestRegressor(n_jobs=-1, random_state=random_state)
param_grid={'max_depth':[int(np.sqrt(X_train_val.shape[1])), 50, 100], 'min_samples_leaf': [1, 5, 10]}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=cv_rs)

gs_r = GridSearchCV(regression_forest,
                  param_grid=param_grid,
                  cv=cv, 
                  verbose=5, 
                  n_jobs=-1, 
                  scoring='r2')

gs_r.fit(X_train_val, y_train_val)

pd.DataFrame(gs_r.cv_results_).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/pca_study/outputs/sample_a/regression_fine_tuning.csv')

#################

classification_forest = RandomForestClassifier(n_jobs=-1, random_state=random_state)
param_grid={'max_depth':[int(np.sqrt(X_train_val.shape[1])), 50, 100], 'min_samples_leaf': [1, 5, 10]}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=cv_rs)

gs_r = GridSearchCV(classification_forest,
                  param_grid=param_grid,
                  cv=cv, 
                  verbose=5, 
                  n_jobs=-1, 
                  scoring='accuracy')

gs_r.fit(X_train_val, y_train_val)

pd.DataFrame(gs_r.cv_results_).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/pca_study/outputs/sample_a/classification_fine_tuning.csv')
####