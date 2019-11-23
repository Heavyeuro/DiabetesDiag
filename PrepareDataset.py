import pandas as pd


from sklearn.impute import SimpleImputer

def SimpleImputingData(X_train, X_valid):


    simple_imputer = SimpleImputer(strategy='mean')#most_frequent')
    imputed_X_train = pd.DataFrame(simple_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(simple_imputer.transform(X_valid))
    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    return imputed_X_train,imputed_X_valid


#def SimpleImputingData(X_train,X_valid):