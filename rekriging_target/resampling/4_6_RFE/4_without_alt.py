import RFE
from importlib import reload
reload(RFE)
from RFE import *
from math import sqrt
from sklearn.metrics import accuracy_score

random_state = 92656

X_train_val = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/model_input/X_train_val.csv')
y_train_val = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/model_input/y_train_val.csv')

classification_logistic = LogisticRegression(n_jobs=-1, random_state=random_state, max_iter=300)
regression_linear = LinearRegression(n_jobs=-1)
regression_forest = RandomForestRegressor(n_jobs=-1, random_state=random_state, max_depth=int(sqrt(X_train_val.shape[1])))
classification_forest = RandomForestClassifier(n_jobs=-1, random_state=random_state, max_depth=int(sqrt(X_train_val.shape[1])))

for m, model in tqdm(enumerate([regression_linear, regression_forest])):

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

    
    explanatory = X_train_val
    target =  y_train_val
    X_train, X_val, y_train, y_val = train_test_split(explanatory,
                                                target,
                                                test_size=0.15,
                                                shuffle=True,
                                                random_state=random_state,
                                                stratify=target)
    results_rfe, rfe_tracking = RecursiveFeatureSelection(X_train, X_val, y_train, y_val, model=model,
                                                            c_y_train = False, c_y_val = False)
    results_rfe.to_csv(f'/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/resampling/outputs/rfe_without_alt/{string2}_{string3}_rfe.csv')
    rfe_tracking.to_csv(f'/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/resampling/outputs/rfe_without_alt/{string2}_{string3}_rfe_tracking.csv')

