import pandas as pd


def readCSV(name):
    return pd.read_csv('files/' + name, encoding='utf-8')


# print a summary of the data
def describeData(CSVfliveName):
    pd.set_option('display.max_columns', 10)
    obj_to_describe = readCSV(CSVfliveName)
    print(obj_to_describe.describe())


def detectNullVal(CSVfliveName):
    obj_to_describe = readCSV(CSVfliveName)
    # missing values by columns
    missing_val_count_by_column = (obj_to_describe.isin([0]).sum())
    print(missing_val_count_by_column)
