import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def convertXLSToCSV(name_CSV, name_xls):
    name_CSV = 'files/'+name_CSV
    data_xls = pd.read_excel('files/'+name_xls, 'Sheet1')#, index_col=None)
    data_xls.to_csv(name_CSV, encoding='utf-8', index= None)

    refactoringCol(name_CSV)


def refactoringCol(name_CSV):
    data_csv = pd.read_csv(name_CSV,)
    for i in range(0, len(data_csv['Diabetes'])):
        if data_csv['Diabetes'][i] == "Healthy":
            data_csv.at[i, 'Diabetes'] = True
        else:
            data_csv.at[i, 'Diabetes'] = False
    data_csv.to_csv(name_CSV, encoding='utf-8', index= None)

def readCSV(name):
    return pd.read_csv('files/'+name, encoding='utf-8')


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=150, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

if __name__ == '__main__':
    nameXLS = 'Diabetes.xls'
    nameCSV = 'csvData.csv'
    convertXLSToCSV(nameCSV, nameXLS)

    X_full=readCSV(nameCSV )

    y=X_full.Diabetes
    X=X_full.drop(['Diabetes'], axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.85, test_size=0.15, random_state=0)
    print(score_dataset(X_train, X_valid, y_train, y_valid))





    # #missing values by columns
    # missing_val_count_by_column = (X.isin([0]).sum())
    # print(missing_val_count_by_column)
