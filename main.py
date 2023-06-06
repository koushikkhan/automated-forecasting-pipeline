# -*- coding: utf-8 -*-

"""
Driver Program for the application

author : Koushik Khan
"""

import os
import pandas as pd
import warnings
import logging
from configparser import ConfigParser
from sklearn.pipeline import Pipeline
from api_utils.api_utils import *
from pipeline.data_transformer import *
from model_utils.model_utils import *

warnings.filterwarnings("ignore")


# define paths
root_dir = os.path.dirname(os.path.realpath(__file__))
conf_dir = os.path.join(root_dir, "config")
output_dir = os.path.join(root_dir, "output")
logs_dir = os.path.join(root_dir, "logs")

# setup logs
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'application.log'), mode="a+"),
        logging.StreamHandler()
    ],
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# read conf file
conf = ConfigParser()
conf.read(os.path.join(conf_dir, "config.ini"))
# print(conf["api"]["base_url"])


def main():
    
    # step: 1 - acquire data from api
    try:
        # get api data and parse it for time series analysis
        api_output = get_api_data_as_json(
            country_code=conf["api"]["country_code"], 
            indicator=conf["api"]["indicator"],
            base_url=conf["api"]["base_url"], 
            max_page=int(conf["api"]["max_page"])
        )
        logging.info("successfully extracted data from api")

        data_ts = parse_api_output_for_tsa(
            api_output=api_output, 
            select_cols=['date', 'value'], 
            sort_by_date=True, 
            set_date_idx=True
        )
        logging.info(f"successfully parsed api output, found {data_ts.shape[0]} records")

        # save data
        # data_ts.to_csv(os.path.join(output_dir, "data", f'api_extract_{conf["api"]["country_code"]}_{conf["api"]["indicator"]}.csv'), index=False)
        # print(data_ts.shape)
    except Exception as e:
        print(str(e))


    # step: 2 - use data pipeline to transform the data
    try:
        ts_pipeline = Pipeline(
            [
                ('imputation', ProcessData('value', 'spline'))
            ]
        )
        transformed_df = ts_pipeline.transform(data_ts)
        logging.info("successfully applied data transformation")
    except Exception as e:
        print(str(e))


    # step: 3 - build model
    try:
        ##### build arima model #####
        logging.info("initializing ARIMA model")
        arima_model = ArimaModel(transformed_df, 'y')
        
        arima_model.get_best_model(train_perc=0.9)
        summary = arima_model.get_model_summary(auto_arima=True)
        logging.info("started building ARIMA model")
        logging.info("printing summary for auto_arima on console")
        print(summary)
        
        logging.info("starting modelling by arima")
        arima_model.build_model(path_to_save=os.path.join(output_dir, "model"))
        summary = arima_model.get_model_summary(auto_arima=False)
        logging.info("built best ARIMA model successfully")
        logging.info("printing summary for best model on console")
        print(summary)

        result_arima = arima_model.forecast_to_df(steps=int(conf['model']['forecast_period']), include_history=True)
        logging.info("forecasts have been compiled successfully for ARIMA")

        mape_arima = compute_mape(result_arima, 'y', 'yhat_arima')
        logging.info(f"mape for ARIMA reported as {mape_arima:.4f} %")

        ##### build prophet model #####
        logging.info("initializing forecasting model using FbProphet")
        prophet_model_test = ProphetModel(transformed_df, 'y')
        
        logging.info("starting modelling by fbprophet")
        prophet_model_test.build_model(path_to_save=os.path.join(output_dir, "model"))
        logging.info("built prophet model successfully")

        result_prophet = prophet_model_test.forecast_to_df(steps=int(conf['model']['forecast_period']), include_history=True)
        logging.info("forecasts have been compiled successfully for prophet")

        mape_prophet = compute_mape(result_prophet, 'y', 'yhat_prophet')
        logging.info(f"mape for prophet model reported as {mape_prophet:.4f} %")

        df_forecast_final = combine_forecasts(
            conf=conf, 
            result_arima=result_arima, 
            result_prophet=result_prophet, 
            path_to_save=os.path.join(output_dir, "data"))
        logging.info("forecasts are combined")

        plot_actual_vs_forecast(
            df_forecast_final, 
            yhat_arima=True, 
            yhat_prophet=True, 
            yhat_ensemble=True, 
            path_to_save=os.path.join(output_dir, "figures")
        )
        logging.info("visualization on actual vs forecast comparison is saved into output/figures")

    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    main()