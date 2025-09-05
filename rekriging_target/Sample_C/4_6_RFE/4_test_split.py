import pandas as pd
from sklearn.model_selection import train_test_split

random_state = 684

data = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/cache/processed_not_imputed.csv')


explanatory = data.loc[:, [col for col in data.columns if col not in ['target', 'fire_id', 'geometry', 'ignition_point (epsg=3857)s', 'YYYY']]]
target =  data.loc[:, ['target']]
X_train, X_test, y_train, y_test = train_test_split(explanatory,
                                                    target,
                                                    test_size=0.15,
                                                    shuffle=True,
                                                    random_state=random_state,
                                                    stratify=target)
pd.DataFrame(X_test).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/model_input/X_test.csv', index=False)
pd.DataFrame(y_test).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/model_input/y_test.csv', index=False)

pd.DataFrame(X_train).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/model_input/X_train_val.csv', index=False)
pd.DataFrame(y_train).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/model_input/y_train_val.csv', index=False)