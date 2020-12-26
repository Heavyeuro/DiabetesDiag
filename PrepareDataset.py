import DataAnalysis as da
import CoreActions as ca
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def buildMLModel(nameCSV):
    X_full = da.readCSV(nameCSV)
    beside_list = ['Pregnancies']
    y = X_full.Diabetes
    X = ca.null_to_NaN(X_full.drop(['Diabetes'], axis=1), beside_list)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.85, test_size=0.15)

    my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
    my_model.fit(X_train, y_train,
                 early_stopping_rounds=5,
                 eval_set=[(X_valid, y_valid)],
                 verbose=False)
    predictions = my_model.predict(X_valid)
    print("Mean Absolute Error: "+str(mean_absolute_error(predictions,y_valid)))
