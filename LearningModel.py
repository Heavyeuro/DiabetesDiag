import PrepareDataset as pds
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score
import DataAnalysis as da
import CoreActions as ca

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

    #ca.null_to_NaN(X, beside_list)
    # si=SimpleImputer(strategy='most_frequent')
    # si.fit(X,y)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=1)

    # results = {}
    # for i in range(1, 10):
    #     results[25 * i] = get_score(25 * i,  X_train, y_train)
    # n_estimators_best = min(results, key=results.get)
    # print(n_estimators_best)
    n_estimators_best=70
    print(get_score(n_estimators_best, X_train, y_train,X_valid,y_valid))


def get_score(n_estimators,X,y,X_valid,y_valid):
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer(strategy='most_frequent')),
        ('model', RandomForestRegressor(n_estimators))
    ])
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')
    my_pipeline.fit(X, y)

    ## Preprocessing of validation data, get predictions
    # preds = my_pipeline.predict(X_valid)
    print(scores)
    # print('MAE:', mean_absolute_error(y_valid, preds))

    import numpy as np
    from sklearn import datasets
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV

    alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
    # create and fit a ridge regression model, testing each alpha
    model = Ridge()
    grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    grid.fit(X, y)
    print(grid)
    # summarize the results of the grid search
    print(grid.best_score_)

    return scores.mean()