from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from numpy import mean
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


class Score:
    """
    mse, r square, mae -> with K_fold
    """
    def __init__(self):
      self.mse_train_list = []
      self.r2_score_train_list = []
      self.mae_train_list = []

      self.mse_test_list = []
      self.r2_score_test_list = []
      self.mae_test_list = []

    def calculate_scores(self, y_train, y_test, y_train_predict, y_test_predict):
        assert len(y_train) == len(y_train_predict) and len(y_test) == len(y_test_predict), "All y don't have the same size"

        self.mse_train_list.append(mean_squared_error(y_train, y_train_predict))
        self.r2_score_train_list.append(r2_score(y_train, y_train_predict))
        self.mae_train_list.append(mean_absolute_error(y_train, y_train_predict))

        self.mse_test_list.append(mean_squared_error(y_test, y_test_predict))
        self.r2_score_test_list.append(r2_score(y_test, y_test_predict))
        self.mae_test_list.append(mean_absolute_error(y_test, y_test_predict))

        self.y  = pd.Series(np.concatenate((y_train, y_test)), name="observation")
        self.y_pred = pd.Series(np.concatenate((y_train_predict, y_test_predict)), name="prediction")
        return self

    def get_score_df(self):
        d = {"mse": [mean(self.mse_train_list), mean(self.mse_test_list)],
             # "r_square": [mean(self.r2_score_train_list), mean(self.r2_score_test_list)],
             "mae": [mean(self.mae_train_list), mean(self.mae_test_list)],
             }
        df = pd.DataFrame(data=d, index=["train", "test"])
        return df

    def get_plot(self):
        #ax = sns.regplot(x=self.y, y=self.y_pred, marker="+")
        d = np.arange(0, 4, 0.1)
        plt.plot(d,d, color="blue", label="perfect model")
        plt.scatter(self.y, self.y_pred)
        plt.xlabel("Observation")
        plt.ylabel("prediction")

        regr = linear_model.LinearRegression()
        regr.fit(np.reshape(self.y.to_numpy(), (-1, 1)), self.y_pred)
        plot = regr.predict(np.reshape(self.y.to_numpy(), (-1, 1)))
        plt.plot(self.y, plot, c="red", label="real model")
        plt.legend(loc="upper left")
