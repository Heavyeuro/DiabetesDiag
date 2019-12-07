import pandas as pd
import DataAnalysis as da
import CoreActions as ca
from sklearn.impute import SimpleImputer
import numpy as np

# Converting XLS file to CSV
def convertXLSToCSV(name_CSV, name_xls):
    name_CSV = 'files/' + name_CSV
    data_xls = pd.read_excel('files/' + name_xls, 'Sheet1')
    data_xls.to_csv(name_CSV, encoding='utf-8', index=None)
    refactoringCol(name_CSV)

# Refactoring output data col to True and False
# Can be replaced in future by Label Encoding for categorical variable
def refactoringCol(name_CSV):
    data_csv = pd.read_csv(name_CSV, encoding='utf-8')
    replaceValInCols(data_csv, ['Diabetes'], 'Healthy', True)
    replaceValInCols(data_csv, ['Diabetes'], 'Sick', False)

    data_csv.to_csv(name_CSV, encoding='utf-8', index=None)

# Replace each val in appropriate cols
def replaceValInCols(data_csv,listColumn, oldValue, newValue):
    for col in listColumn:
         data_csv[col]=data_csv[col].replace(oldValue, newValue)
    return data_csv

# Replacing missing values (imputing) according to certain strategy
def SimpleImputingData(X_train, X_valid):
    simple_imputer = SimpleImputer(strategy='most_frequent')
    imputed_X_train = pd.DataFrame(simple_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(simple_imputer.transform(X_valid))
    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    return imputed_X_train,imputed_X_valid


# replacing 0 to NaN in dataset
def null_to_NaN(X, except_list):
    cols_with_mising_val = da.detectNullVal(X, except_list).index[0]
    X = ca.replaceValInCols(X, [cols_with_mising_val], 0, np.nan)
    return X