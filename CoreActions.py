import pandas as pd


def convertXLSToCSV(name_CSV, name_xls):
    name_CSV = 'files/' + name_CSV
    data_xls = pd.read_excel('files/' + name_xls, 'Sheet1')
    data_xls.to_csv(name_CSV, encoding='utf-8', index=None)

    refactoringCol(name_CSV)


def refactoringCol(name_CSV):
    data_csv = pd.read_csv(name_CSV, )
    for i in range(0, len(data_csv['Diabetes'])):
        if data_csv['Diabetes'][i] == "Healthy":
            data_csv.at[i, 'Diabetes'] = True
        else:
            data_csv.at[i, 'Diabetes'] = False
    data_csv.to_csv(name_CSV, encoding='utf-8', index=None)
