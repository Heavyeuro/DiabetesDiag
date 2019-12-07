import pandas as pd

# reading file in CSV format
def readCSV(name):
    return pd.read_csv('files/' + name, encoding='utf-8')


# print a summary of the data
def describeData(CSVfliveName):
    pd.set_option('display.max_columns', 10)
    obj_to_describe = readCSV(CSVfliveName)
    obj_to_describe.describe()

# Counting an amount of null values in dataset by coloumn
def detectNullVal(obj_to_describe, exclude_col=[]):
    # missing values by columns
    obj_to_describe=obj_to_describe.drop(exclude_col, axis=1)
    missing_val_count_by_column = (obj_to_describe.isin([0]).sum())
    return (missing_val_count_by_column)
