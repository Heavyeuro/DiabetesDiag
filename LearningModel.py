import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import DataAnalysis as da


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=150, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


def buildMLModel(nameCSV):
    X_full = da.readCSV(nameCSV)

    y = X_full.Diabetes
    X = X_full.drop(['Diabetes'], axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.85, test_size=0.15, random_state=0)

    my_imputer = SimpleImputer()

    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid, ))

    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
    print(score_dataset(X_train, X_valid, y_train, y_valid))
