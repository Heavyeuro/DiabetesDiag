import CoreActions as ca
import DataAnalysis as da
import LearningModel as lm
import PrepareDataset as pds

if __name__ == '__main__':
    nameXLS = 'Diabetes.xls'
    nameCSV = 'csvData.csv'

    ca.convertXLSToCSV(nameCSV, nameXLS)

    da.describeData(nameCSV)
    #da.detectNullVal(nameCSV)
    pds.buildMLModel(nameCSV)

    lm.buildMLModel(nameCSV)
