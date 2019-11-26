import pandas as pd
import DataAnalysis as da
import CoreActions as ca
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def buildMLModel(nameCSV):
    X_full = da.readCSV(nameCSV)
    beside_list = ['Pregnancies']
    y = X_full.Diabetes
    X = ca.null_to_NaN(X_full.drop(['Diabetes'], axis=1), beside_list)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=1)
    #X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    # my_model = XGBRegressor()
    # my_model.fit(X_train, y_train)

    my_model = XGBRegressor(n_estimators=500, learning_rate=0.06, n_jobs=2)
    my_model.fit(X_train, y_train,
                 early_stopping_rounds=5,
                 eval_set=[(X_valid, y_valid)],
                 verbose=False)
    predictions = my_model.predict(X_valid)
    print("Mean Absolute Error: "+str(mean_absolute_error(predictions,y_valid)))

   #  beside_list = ['Pregnancies']
   #  ca.null_to_NaN(X, beside_list)
   #
   #  my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer(strategy='most_frequent')),
   #                                ('model', RandomForestRegressor(n_estimators=450, random_state=0))])
   #
   #  # Multiply by -1 since sklearn calculates *negative* MAE
   #
   #  X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=1)
   # # X_train, X_valid=pds.SimpleImputingData(X_train, X_valid)
   #
   #  results = {}
   #  for i in range(1, 10):
   #      results[25 * i] = get_score(25 * i,  X_train, y_train)
   #  n_estimators_best = min(results, key=results.get)
   #  print(get_score(n_estimators_best, X_train, y_train))