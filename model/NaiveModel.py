from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from numpy import mean
from .Score import Score
import pandas as pd


class NaiveModel:
    """
    Predit mean
    """
    def __init__(self):
        self.cv = 10

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

            s.calculate_scores(y_train, y_test, [mean(y_train)] * len(y_train), [mean(y_train)] * len(y_test))
            break
        return s





