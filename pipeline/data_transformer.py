# -*- coding: utf-8 -*-

"""
Custom data transformation pipelines

author : Koushik Khan
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.stattools import adfuller


class ProcessData(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for pre-processing time series data
    """
    def __init__(self, col_name, impute_method):
        self.col_name = col_name
        self.impute_method = impute_method
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame), "transform() expects a pandas DataFrame object!"

        if X[self.col_name].isna().any():
            if self.impute_method == "mean":
                X = X.assign(target=X[self.col_name].fillna(X[self.col_name].mean()))
            elif self.impute_method == "median":
                X = X.assign(target=X[self.col_name].fillna(X[self.col_name].median()))
            elif self.impute_method == "locf":
                X = X.assign(target=X[self.col_name].fillna(method ='bfill'))
            elif self.impute_method == "nocb":
                X = X.assign(target=X[self.col_name].fillna(method ='ffill'))
            elif self.impute_method == "linear":
                X = X.assign(target=X[self.col_name].interpolate(method="linear"))
            elif self.impute_method == "spline":
                X = X.assign(target=X[self.col_name].interpolate(option="spline"))
            else:
                raise(ValueError("method argument is not having a permissible value"))
        
        return X
    

class MakeStationary(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for processing time series data
    """
    def __init__(self, col_name, adf_alpha):
        self.col_name = col_name
        self.adf_alpha = adf_alpha

    @staticmethod
    def check_stationarity(df, col_name, alpha=0.05):
        # extract series
        series = df[col_name]

        # drop na
        if series.isna().any():
            series.dropna(axis=0, inplace=True)
        
        # perform adf test
        test_statistic, p_value, _, _, _, _ = adfuller(series)

        # compare p-value with significance level
        if p_value < alpha:
            # reject the null hyp. --> the series is actually stationary
            return test_statistic, p_value, True
        else:
            # accept the null hyp. --> the series is actually non-stationary 
            return test_statistic, p_value, False
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame), "transform() expects a pandas DataFrame object!"

        # check for stationarity
        _, _, is_stationary = self.check_stationarity(X, self.col_name, self.adf_alpha)
        # print(f"is stationary: {is_stationary}")

        if not is_stationary:
            # take difference to make the series stationary
            # X['diff_target'] = X['target'].diff()
            X = X.assign(diff_target = X[self.col_name].diff())

        return X


if __name__ == "__main__":
    # define tests
    pass