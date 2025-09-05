import RFE
from importlib import reload
reload(RFE)
from RFE import *
from math import sqrt
from sklearn.metrics import accuracy_score

random_state = 92656

y_train_val = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input_resampled/y_train_val.csv').drop(columns='fire_id')
X_pca = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input_resampled/pca_tranformed_without_target.csv')


classification_logistic = LogisticRegression(n_jobs=-1, random_state=random_state, max_iter=300)
regression_linear = LinearRegression(n_jobs=-1)
regression_forest = RandomForestRegressor(n_jobs=-1, random_state=random_state, max_depth=int(sqrt(X_pca.shape[1])))
classification_forest = RandomForestClassifier(n_jobs=-1, random_state=random_state, max_depth=int(sqrt(X_pca.shape[1])))

for m, model in tqdm(enumerate([regression_linear, regression_forest, classification_logistic, classification_forest])):

    if m < 2:
        string2='regression'
        y='target'
        metric=adj_r2
    else:
        string2 = 'classification'
        y='label'
        metric=accuracy_score
    if (m==0) | (m==2):
        string3 = 'lr'
    else:
        string3 = 'rf'

    
    explanatory = X_pca
    target =  y_train_val
    X_train, X_val, y_train, y_val = train_test_split(explanatory,
                                                target,
                                                test_size=0.15,
                                                shuffle=True,
                                                random_state=random_state,
                                                stratify=target)
    results_rfe, rfe_tracking = RecursiveFeatureSelection(X_train, X_val, y_train, y_val, model=model,
                                                            c_y_train = False, c_y_val = False)
    results_rfe.to_csv(f'/nfs/home/genovese/thesis-wildfire-genovese/resampling/outputs/rfe_pca/{string2}_{string3}_rfe.csv')
    rfe_tracking.to_csv(f'/nfs/home/genovese/thesis-wildfire-genovese/resampling/outputs/rfe_pca/{string2}_{string3}_rfe_tracking.csv')

