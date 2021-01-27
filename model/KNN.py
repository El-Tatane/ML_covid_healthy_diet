from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import mean_squared_error
from numpy import mean
from .Score import Score


class KNN:

    def __init__(self, cv=10):
        self.cv = cv

    def fit_one(self, X, y, n_neighbors=5):
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

            self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
            self.model.fit(X_train, y_train)

            s.calculate_scores(y_train, y_test, self.model.predict(X_train), self.model.predict(X_test))
        return s

    def fit(self, X, y, min_k=3, max_k=15):
        res = {}
        for i in range(min_k, max_k + 1):
            score = self.fit_one(X, y, n_neighbors=i)
            res[i] = score

        return res