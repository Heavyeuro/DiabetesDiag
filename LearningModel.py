import PrepareDataset as pds
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score
import DataAnalysis as da


def score_dataset(X_train, X_valid, y_train, y_valid, nodes):
    model = RandomForestRegressor(n_estimators=nodes, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# def find_best_num_nodes(X_train, X_valid, y_train, y_valid):
#     next_temp=1
#     for i in range(4, 20):
#         temp=score_dataset(X_train, X_valid, y_train, y_valid, i*25)
#         if(temp<next_temp):
#             next_temp=temp
#             nodes = i*25
#
#     return next_temp, nodes


def buildMLModel(nameCSV):
    X_full = da.readCSV(nameCSV)

    y = X_full.Diabetes
    X = X_full.drop(['Diabetes'], axis=1)

    beside_list = ['Pregnancies']
    pds.null_to_NaN(X, beside_list)

    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer(strategy='most_frequent')),
                                  ('model', RandomForestRegressor(n_estimators=450, random_state=0))])

    # Multiply by -1 since sklearn calculates *negative* MAE

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=1)
   # X_train, X_valid=pds.SimpleImputingData(X_train, X_valid)

    results = {}
    for i in range(1, 10):
        results[25 * i] = get_score(25 * i,  X_train, y_train)
    n_estimators_best = min(results, key=results.get)
    print(get_score(n_estimators_best, X_train, y_train))



def get_score(n_estimators,X,y):
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators, random_state=0))
    ])
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()