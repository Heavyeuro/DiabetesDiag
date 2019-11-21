import CoreActions as ca
import DataAnalysis as da
import LearningModel as lm

if __name__ == '__main__':
    nameXLS = 'Diabetes.xls'
    nameCSV = 'csvData.csv'

    ca.convertXLSToCSV(nameCSV, nameXLS)

    da.describeData(nameCSV)
    da.detectNullVal(nameCSV)

    lm.buildMLModel(nameCSV)
