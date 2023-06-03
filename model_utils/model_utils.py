# -*- coding: utf-8 -*-

"""
This script provides functions to develop time series models
author: Koushik Khan
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from pmdarima.arima import ARIMA
from pmdarima.arima import auto_arima, ADFTest


def train_test_split(data, response_col, training_perc = 0.8):
    df_with_response_col = data[[response_col]]
    split_idx = round(len(df_with_response_col) * training_perc)
    train = df_with_response_col.iloc[:split_idx]
    test = df_with_response_col.iloc[split_idx:]
    return df_with_response_col, train, test


def visualize_train_test_split(train_df, test_df, path_to_save_fig, file_name):
    fig, ax= plt.subplots(figsize=(14,10))
    kws = dict(marker='o')
    plt.plot(train_df, label = 'Train', **kws)
    plt.plot(test_df, label = 'Test', **kws)
    ax.legend(bbox_to_anchor=[1,1])
    plt.savefig(os.path.join(path_to_save_fig, file_name))


def get_best_arima_model(train_df, **kwargs):
    arima_model = auto_arima(train_df, start_p=0, start_q=0, **kwargs)
    return [arima_model.order, arima_model.seasonal_order]


def fit_best_arima_model(data, best_model_params, path_to_save_model):
    best_model = ARIMA(
        order=best_model_params[0],
        seasonal_order=best_model_params[1]
    ).fit(data)

    # save model
    with open(os.path.join(path_to_save_model, 'arima.pkl'), 'wb') as pkl:
        pickle.dump(best_model, pkl)

    return best_model


def visualize_model_diagnostics():
    pass


def get_forecast_in_dataframe(path_to_save_model, steps):
    # load model
    with open(os.path.join(path_to_save_model, 'arima.pkl'), 'rb') as pkl:
        model = pickle.load(pkl)

    preds, conf_int = model.predict(n_periods=steps, return_conf_int=True)
    pred_df = pd.DataFrame({"pred":preds})
    pred_df['lower'] = conf_int[:,0]
    pred_df['upper'] = conf_int[:,1]
    return pred_df


def visualize_prediction(train_df, test_df, pred_df, title, path_to_save_fig, file_name):
    fig,ax = plt.subplots(figsize=(14,10))
    kws = dict(marker='o')
    
    ax.plot(train_df,label='Train',**kws)
    ax.plot(test_df,label='Test',**kws)
    ax.plot(pred_df['pred'], label='Prediction', ls='--',linewidth=3)

    ax.fill_between(x=pred_df.index,y1=pred_df['lower'], y2=pred_df['upper'], alpha=0.3)
    ax.set_title(title, fontsize=22)
    ax.legend(loc='upper left')
    fig.tight_layout()
    
    plt.savefig(os.path.join(path_to_save_fig, file_name))


def generate_output(hist_df, forecast_df, path_to_save_op, file_name):
    forecast_with_history_df = pd.concat([hist_df, forecast_df])

    forecast_with_history_df.reset_index(inplace=True)
    forecast_with_history_df.rename(columns={"index":"date"}, inplace=True)

    forecast_with_history_df.to_csv(os.path.join(path_to_save_op, file_name), index=True)
