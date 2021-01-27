from scipy import stats
import pandas as pd
from sklearn.metrics import r2_score
from math import sqrt
from scipy.stats import t, sem
from numpy import mean


class StudentTest:

    def __init__(self, df, output_col, alpha=0.05):
        self.df = df.copy()
        self.output_col = output_col
        self.alpha = alpha

    def fit(self):
        """
        return df with only variable with correlation after student test
        :return:
        """
        ret_col_list = []
        for col in self.df.columns:
            if col == self.output_col:
                ret_col_list.append(col)
            else:
                if self.is_dependant(col, self.output_col):
                    ret_col_list.append(col)
        return ret_col_list, self.df[ret_col_list]

    def independent_ttest(self, data1, data2, alpha):
        mean1, mean2 = mean(data1), mean(data2)
        # standard errors
        se1, se2 = sem(data1), sem(data2)
        # standard error on the difference between the samples
        sed = sqrt(se1**2.0 + se2**2.0)
        # calculate the t statistic
        t_stat = (mean1 - mean2) / sed
        # degrees of freedom
        df = len(data1) + len(data2) - 2
        # calculate the critical value
        cv = t.ppf(1.0 - alpha, df)
        # calculate the p-value
        p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
        # return everything
        return t_stat, df, cv, p

    def is_dependant(self, x_col, y_col):
        t_stat, df, cv, p = self.independent_ttest(self.df[x_col], self.df[y_col], self.alpha)

        if abs(t_stat) <= cv:
            # print('Accept null hypothesis that the means are equal.')
            return False
        else:
            return True
            # print('Reject the null hypothesis that the means are equal.')