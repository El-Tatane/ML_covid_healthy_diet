from sklearn.svm import SVR
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from numpy import mean
from .Score import Score


class SVM:

    def __init__(self, cv=10):
        self.cv = cv

    def fit(self, X, y):
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

            self.model = SVR()
            self.model.fit(X_train, y_train)
            s.calculate_scores(y_train, y_test, self.model.predict(X_train), self.model.predict(X_test))
        return s


