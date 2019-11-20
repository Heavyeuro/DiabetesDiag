import pandas as pd

def convertToCSV():
    name_CSV = 'files/csvfile.csv'
    data_xls = pd.read_excel('files/Diabetes.xls', 'Sheet1', index_col=None)
    data_xls.to_csv(name_CSV, encoding='utf-8')
    refactoringCol(name_CSV)


def refactoringCol(name_CSV):
    data_csv = pd.read_csv(name_CSV, encoding='utf-8')

    i=0
    while i < len(data_csv['Diabetes']):
        if data_csv['Diabetes'][i] == "Healthy":
            data_csv.at[i, 'Diabetes'] = True
        else:
            data_csv.at[i, 'Diabetes'] = False
        i = i+1

    data_csv.to_csv(name_CSV)


if __name__ == '__main__':
    convertToCSV()
    print(pd.read_csv('files/csvfile.csv', encoding='utf-8'))