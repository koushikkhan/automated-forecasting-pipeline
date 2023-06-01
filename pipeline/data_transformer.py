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

# ------------ Set Path ------------- #
home_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(home_path, "data")
algo_path = os.path.join(home_path, "algo")
model_path = os.path.join(home_path, "model")
sys.path.append(home_path)

# lemmatizer = WordNetLemmatizer()

class CleanTextData(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for cleaning text data

    # Arguments
        col_name: column name of the source dataframe
        min_word_len: minimum length of the word to be retained
        stop: English stop-word list from sklearn.feature_extraction.text
        lemmatizer: lemmatizer to be used, default is WordnetLemmatizer
    """
    def __init__(self, col_name, min_word_len=3, stop=list(text.ENGLISH_STOP_WORDS), lemmatizer=WordNetLemmatizer()):
        self.col_name = col_name
        self.min_word_len = min_word_len
        self.stop = stop
        self.lemmatizer = lemmatizer

    def clean_text(self, text):
        """
        # Arguments
            text: text body to be preprocessed and cleaned

        # Return
            cleaned text
        """
        # handle non-ascii/special characters
        text = text.encode("utf-8")
        text = re.sub(r"\\[ux][a-z0-9]+", " ", str(text))
        text = str(text).replace("b", "")
        text = text.strip("'").lower()
        text = re.sub(r'[\:\-\(\)\%\d\.\\\/\_\[\]\+\,\#\"]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        word_list = text.split(' ') # tokenization w.r.t space characters
        rel_words = [word for word in word_list if word not in self.stop and len(word) >= self.min_word_len] # relevant words
        rel_words_lemm = [self.lemmatizer.lemmatize(word, pos='v') for word in rel_words]
        return " ".join(rel_words_lemm)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame), "transform() expects a pandas DataFrame object!"
        return X[self.col_name].apply(self.clean_text)


class CreateDummies(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for creating dummy variables for
    categorical data

    # Arguments
        col_name: column name of the source dataframe (categorical in nature)
    """
    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        # Arguments
            pandas dataframe

        # Returns
            pandas dataframe object

        # References
            1. https://stackoverflow.com/questions/45703590/duplicating-pandas-get-dummies-columns-from-train-to-test-data
            2. https://stackoverflow.com/questions/37425961/dummy-variables-when-not-all-categories-are-present/37451867#37451867
        """
        return pd.get_dummies(X[self.col_name].astype(pd.api.types.CategoricalDtype(categories = ["DISK", "PORT", "RAM", "None"])))
        # return pd.get_dummies(X[self.col_name])
        # return d


def main():
    print("Test")
    return None


if __name__ == "__main__":
    main()