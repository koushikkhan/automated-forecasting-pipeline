# -*- coding: utf-8 -*-

"""
Custom data transformation pipelines

author : Koushik Khan
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.feature_extraction import text
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_selection import chi2, f_classif, SelectKBest


class ProcessData(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for processing time series data
    """
    def __init__(self, col_name, method):
        self.col_name = col_name
        self.method = method
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame), "transform() expects a pandas DataFrame object!"

        if self.method == "mean":
            X = X.assign(target=X[self.col_name].fillna(X[self.col_name].mean()))
        elif self.method == "median":
            X = X.assign(target=X[self.col_name].fillna(X[self.col_name].median()))
        elif self.method == "locf":
            X = X.assign(target=X[self.col_name].fillna(method ='bfill'))
        elif self.method == "nocb":
            X = X.assign(target=X[self.col_name].fillna(method ='ffill'))
        elif self.method == "linear":
            X = X.assign(target=X[self.col_name].interpolate(method="linear"))
        elif self.method == "spline":
            X = X.assign(target=X[self.col_name].interpolate(option="spline"))
        else:
            raise(ValueError("method argument is not having a permissible value"))
        
        return X
    

class MakeStationary(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for making time series stationary
    """
    def __init__(self, col_name):
        self.col_name = col_name

    def make_stationary(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pass


def main():
    print("Test")
    return None


if __name__ == "__main__":
    main()