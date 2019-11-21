import pandas as pd
from sklearn.model_selection import train_test_split

def convertXLSToCSV(name_CSV, name_xls):
    name_CSV = 'files/'+name_CSV
    data_xls = pd.read_excel('files/'+name_xls, 'Sheet1', index_col=None)
    data_xls.to_csv(name_CSV, encoding='utf-8')
    refactoringCol(name_CSV)


def refactoringCol(name_CSV):
    data_csv = pd.read_csv(name_CSV, encoding='utf-8')

    for i in range(0, len(data_csv['Diabetes'])):
        if data_csv['Diabetes'][i] == "Healthy":
            data_csv.at[i, 'Diabetes'] = True
        else:
            data_csv.at[i, 'Diabetes'] = False
    data_csv.to_csv(name_CSV)

def readCSV(name):
    return pd.read_csv('files/'+name)#, index_col='Id')


if __name__ == '__main__':
    nameXLS = 'Diabetes.xls'
    nameCSV = 'csvData.csv'
    convertXLSToCSV(nameCSV, nameXLS)

    X_full=readCSV(nameCSV )
    print(X_full)
