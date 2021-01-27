from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from numpy import mean
import pandas as pd
from .Score import Score


class HyperOptimizedRandomForest:

    def __init__(self, param_grid=None, cv=10):

        self.param_grid = {
            'n_estimators': [5, 10, 15, 20, 30, 40],
            'max_depth': [2, 3, 4]
        } if param_grid is None else param_grid
        self.cv = cv

    def fit(self, X, y):
        clf = RandomForestRegressor()
        self.grid_clf = GridSearchCV(clf, self.param_grid, cv=10)
        self.grid_clf.fit(X, y)

    def fit_best_rf(self, X, y):

        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.DataFrame) else y
        y = y.reshape(-1)

        s = Score()
        kf = KFold(n_splits=self.cv)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train = y_train.reshape(-1)
            self.X_train = X_train  # rustine for SHAP


            self.model = RandomForestRegressor(**self.grid_clf.best_params_)
            self.model.fit(X_train, y_train)
            s.calculate_scores(y_train, y_test, self.model.predict(X_train), self.model.predict(X_test))

        return s