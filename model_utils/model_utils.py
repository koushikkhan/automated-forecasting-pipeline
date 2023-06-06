# -*- coding: utf-8 -*-

"""
This script provides functions to develop time series models
author: Koushik Khan
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from pmdarima.arima import ARIMA
from pmdarima.arima import auto_arima, ADFTest


##### Classes for developing models #####
class ArimaModel:
    """
    Helper class to build ARIMA model using auto_arima  
    """
    def __init__(self, data, target_col):
        self.data = data
        self.target_col = target_col
        self.auto_arima_model = None
        self.best_model = None

    def get_best_model(self, train_perc, path_to_save=None, **kwargs):
        # split data into train and test
        temp_data = self.data[[self.target_col]]

        split_idx = round(len(temp_data) * train_perc)

        # split into train and test dataset
        train = temp_data.iloc[:split_idx]
        test = temp_data.iloc[split_idx:]

        # finds best model hyper-parameters and update self.auto_arima_model
        self.auto_arima_model = auto_arima(
            train, start_p=0, start_q=0, **kwargs
        )

        if path_to_save is not None:
            with open(os.path.join(path_to_save, "auto_arima_best_model.pkl")) as pkl:
                pkl.dump(self.auto_arima_model)

    def build_model(self, path_to_save=None):
        if self.auto_arima_model is not None:
            self.best_model = ARIMA(
                order=self.auto_arima_model.order,
                seasonal_order=self.auto_arima_model.seasonal_order
            ).fit(self.data[self.target_col])

            if path_to_save is not None:
                with open(os.path.join(path_to_save, "arima_best_model.pkl"), "wb") as pkl:
                    pickle.dump(self.best_model, pkl)
        else:
            raise(ValueError('please call method .get_best_model() before building best arima model'))

    def forecast_to_df(self, steps, include_history=False):
        # create present df with out-of-sample forecast
        yhat, conf_int = self.best_model.predict(n_periods=steps, return_conf_int=True)
        pred_df = pd.DataFrame({"y":np.nan, "yhat_arima":yhat})
        pred_df['lower_bound_arima'] = conf_int[:,0]
        pred_df['upper_bound_arima'] = conf_int[:,1]
        
        final_df = pred_df

        if include_history:
            # create historical df with in-sample forecast
            hist_yhat, hist_conf_int = self.best_model.predict_in_sample(return_conf_int=True)
            pred_df_hist = pd.DataFrame({"y":self.data['y'].values, "yhat_arima":hist_yhat})
            pred_df_hist['lower_bound_arima'] = hist_conf_int[:,0]
            pred_df_hist['upper_bound_arima'] = hist_conf_int[:,1]

            # combine
            final_df = pd.concat([pred_df_hist, pred_df], axis=0)

        # fix index
        final_df = final_df.reset_index()
        final_df.rename(columns={"index":"ds"}, inplace=True)
        
        return final_df
    
    def get_model_summary(self, auto_arima=False):
        if auto_arima:
            return self.auto_arima_model.summary()
        else:
            return self.best_model.summary()


class ProphetModel:
    """
    Helper class to build model using fbprophet
    """
    def __init__(self, data, target_col):
        self.data = data
        self.target_col = target_col
        self.model = None

    def build_model(self, path_to_save=None):
        # extract target col
        data_temp = self.data[[self.target_col]]

        # small adjustments for prophet
        self.data = self.data.reset_index()
        self.data.rename(columns={"date":"ds"}, inplace=True)

        # define model
        self.model = Prophet()
        self.model.fit(self.data)

        if path_to_save is not None:
            with open(os.path.join(path_to_save, "prophet_model.pkl"), "wb") as pkl:
                pickle.dump(self.model, pkl)

    def forecast_to_df(self, steps, freq="YS", include_history=False):
        future_df = self.model.make_future_dataframe(
            periods=steps, freq=freq, include_history=False
        )
        forecast = self.model.predict(future_df)
        pred_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        pred_df = pred_df.assign(y=np.nan)

        if include_history:
            future_df = self.model.make_future_dataframe(
                periods=steps, freq=freq, include_history=include_history
            )
            forecast = self.model.predict(future_df)
            pred_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

            # insert actual values (y)
            # pred_df = pred_df.assign(y=np.nan)
            pred_df.loc[0:len(self.data), 'y'] = pd.Series(self.data['y'].values)

        final_df = pred_df

        final_df.rename(
            columns={
                "yhat":"yhat_prophet", 
                "yhat_lower":"lower_bound_prophet",
                "yhat_upper":"upper_bound_prophet"
            }, inplace=True
        )

        return final_df
    

# generate visualizations & reports
def combine_forecasts(conf, result_arima, result_prophet, path_to_save):
    df_forecast_final = pd.merge(
        result_arima,
        result_prophet,
        on=['ds', 'y'], 
        how='inner'
    )
    df_forecast_final['yhat_ensemble'] = df_forecast_final[['yhat_arima', 'yhat_prophet']].mean(axis=1)
    
    if path_to_save is not None:
        df_forecast_final.to_csv(
            os.path.join(
                path_to_save, 
                f"forecast_table_{conf['api']['country_code']}_{conf['api']['indicator']}_hist_2022_fc_2030.csv"
            ), index=False
        )

    return df_forecast_final

##### utilities #####
def compute_mape(forecast_table, actual_col, fc_col):
    actuals = forecast_table[actual_col].values
    fcs = forecast_table[fc_col].values

    mape = np.nanmean(np.abs((actuals - fcs)/actuals)) * 100
    return mape


# visualization
def plot_actual_vs_forecast(forecast_df, yhat_arima=False, yhat_prophet=False, yhat_ensemble=False, path_to_save=None):
    # prepare data for visualization
    data_viz = forecast_df.copy()
    data_viz.set_index('ds', inplace=True)

    # plot configurations
    fig,ax = plt.subplots(figsize=(14,10))
    kws = dict(marker='o', color='r')
    ax.plot(data_viz['y'], label="actual", **kws)

    # include series by choice
    if yhat_arima:
        ax.plot(data_viz['yhat_arima'], label='ARIMA', ls='--', linewidth=2)
    
    if yhat_prophet:
        ax.plot(data_viz['yhat_prophet'], label='Prophet', ls='-.', linewidth=2)
    
    if yhat_ensemble:
        ax.plot(data_viz['yhat_ensemble'], label='Ensemble', ls='solid', linewidth=2, color='k')

    plt.grid()

    ax.set_title("Actual vs Forecast Data", fontsize=18)
    ax.legend(loc='upper left')
    fig.tight_layout()

    if path_to_save is not None:
        fig2save = fig.get_figure()
        fig2save.savefig(os.path.join(path_to_save, 'actual_vs_forecast_data.png'))