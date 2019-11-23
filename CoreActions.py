import pandas as pd


def convertXLSToCSV(name_CSV, name_xls):
    name_CSV = 'files/' + name_CSV
    data_xls = pd.read_excel('files/' + name_xls, 'Sheet1')
    data_xls.to_csv(name_CSV, encoding='utf-8', index=None)

    refactoringCol(name_CSV)


def refactoringCol(name_CSV):
    data_csv = pd.read_csv(name_CSV, encoding='utf-8')
    replace(data_csv,['Diabetes'], 'Healthy', True)
    replace(data_csv, ['Diabetes'], 'Sick', False)

    data_csv.to_csv(name_CSV, encoding='utf-8', index=None)


def replace(data_csv,listColumn, oldValue, newValue):
    for col in listColumn:#[0]
         data_csv[col]=data_csv[col].replace(oldValue, newValue)

    return data_csv
