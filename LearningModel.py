import PrepareDataset as pds
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import DataAnalysis as da
import  CoreActions as ca

def score_dataset(X_train, X_valid, y_train, y_valid,nodes):
    model = RandomForestRegressor(n_estimators=nodes, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


def find_best_num_nodes(X_train, X_valid, y_train, y_valid):
    next_temp=1
    for i in range(4, 20):
        temp=score_dataset(X_train, X_valid, y_train, y_valid, i*25)
        if(temp<next_temp):
            next_temp=temp
            nodes=i*25
    return next_temp,nodes


def buildMLModel(nameCSV):
    X_full = da.readCSV(nameCSV)

    y = X_full.Diabetes
    X = X_full.drop(['Diabetes'], axis=1)

    # replacing 0 to NaN
    cols_with_mising_val = da.detectNullVal(X, ['Pregnancies']).index[0]
    X = ca.replace(X, [cols_with_mising_val], 0, np.nan)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=0)
    X_train, X_valid=pds.SimpleImputingData(X_train, X_valid)


    print(find_best_num_nodes(X_train, X_valid, y_train, y_valid))
