import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


input_csv_path = "/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/dataset.csv"

df = pd.read_csv(input_csv_path).drop_duplicates()

non_train_columns = ['target', 'geometry', 'YYYY', 'DD', 'MM', 'day', 'id', 'fire_id']
target_column = 'target'
predictor_columns = [col for col in df.columns if col not in non_train_columns]

test_df = df.sample(frac=0.15, random_state=26165)
test_indices = test_df.index
X_test = test_df.loc[:, predictor_columns]
y_test = test_df.loc[:, target_column]

train_indices = [idx for idx in df.index if idx not in test_indices]
train_df = df.loc[train_indices, predictor_columns + [target_column]]



# leave-one-out cross validation since the dataset is small
loo = LeaveOneOut()
y_true = []
y_pred = []

with open('/nfs/home/genovese/thesis-wildfire-genovese/outputs/random_forest.pkl', 'wb') as f:
    for train_index, test_index in loo.split(train_df):

        X_train, X_test = train_df.loc[train_index, predictor_columns], train_df.loc[test_index, predictor_columns]
        y_train, y_test = train_df[train_index, target_column], train_df.loc[test_index, target_column]

        # Apply MinMax scaling only on training data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest Regressor
        model = RandomForestRegressor(random_state=3469)
        model.fit(X_train_scaled, y_train)

        # Predict on the single test instance
        y_pred.append(model.predict(X_test_scaled)[0])
        y_true.append(y_test[0])


    pickle.dump(model, f)


test_scaler = MinMaxScaler()
X_val_scaled = scaler.fit_transform(train_df.loc[:, predictor_columns])
train_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
train_r2 = r2_score(y_true, y_pred)

test_scaler = MinMaxScaler()
X_test_scaled = scaler.transform(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_scaled)))
test_r2 = r2_score(y_test, model.predict(X_test_scaled))

results = pd.DataFrame({'rmse': {'train': train_rmse, 'test': test_rmse}, 'r2': {'train': train_r2, 'test': test_r2}})
results.to_csv('/nfs/home/genovese/thesis-wildfire-genovese/outputs/random_forest/model_metrics.csv')