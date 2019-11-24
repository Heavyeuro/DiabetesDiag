import pandas as pd
import DataAnalysis as da
import  CoreActions as ca
from sklearn.impute import SimpleImputer
import numpy as np

def SimpleImputingData(X_train, X_valid):

    simple_imputer = SimpleImputer(strategy='most_frequent')#most_frequent')
    imputed_X_train = pd.DataFrame(simple_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(simple_imputer.transform(X_valid))
    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    return imputed_X_train,imputed_X_valid



def null_to_NaN(X,beside_list):
# replacing 0 to NaN
    cols_with_mising_val = da.detectNullVal(X, beside_list).index[0]
    X = ca.replace(X, [cols_with_mising_val], 0, np.nan)