# -*- coding: utf-8 -*-


import os
import sys
import pandas as pd
import warnings
import logging
from configparser import ConfigParser
from sklearn.base import BaseEstimator, TransformerMixin
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
    filename=os.path.join(logs_dir, 'application.log'),
    filemode="a+",
    level=logging.INFO,
    format='%(asctime)s %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p'
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
        logging.info(f"successfully parsed api output, found {data_ts.shape[0]} many records")

        # save data
        # data_ts.to_csv(os.path.join(output_dir, "data", f'api_extract_{conf["api"]["country_code"]}_{conf["api"]["indicator"]}.csv'), index=False)
        # print(data_ts.shape)
    except Exception as e:
        print(str(e))


    # step: 2 - use data pipeline to transform the data
    try:
        ts_pipeline = Pipeline(
            [
                ('imputation', ProcessData('value', 'spline')),
                ('make_stationary', MakeStationary('target', 0.05))
            ]
        )

        transformed_df = ts_pipeline.transform(data_ts)
        logging.info("successfully applied data transformation pipeline")
    except Exception as e:
        print(str(e))


    # step: 3 - build model
    try:
        # split data
        full_data, train, test = train_test_split(
            data=transformed_df, 
            response_col="target", 
            training_perc = 0.8
        )
        logging.info("splitting of data is now complete")

        visualize_train_test_split(
            train_df=train, 
            test_df=test, 
            path_to_save_fig=os.path.join(output_dir, "figures"), 
            file_name="splitted_data.png")
        logging.info("data visualization is ready after splitting the data and saved into output/figures directory")
        
        logging.info("starting auto_arima to find best arima model hyper-parameters")
        # get hyper-parameters for best model using auto_arima
        best_model_params = get_best_arima_model(
            train_df=train
        )
        logging.info("auto_arima process is now complete")

        logging.info("starting the build process for best arima model")
        # fit best model on full data
        best_model = fit_best_arima_model(
            data=full_data,
            best_model_params=best_model_params,
            path_to_save_model=os.path.join(output_dir, "model")
        )
        logging.info("best arima model is now ready and saved into output/model directory")


        # load model and make forecast
        forecast_df = get_forecast_in_dataframe(
            path_to_save_model=os.path.join(output_dir, "model"),
            steps=int(conf["model"]["forecast_period"])
        )
        logging.info(f'forecasts are now ready with period {conf["model"]["forecast_period"]}')

        # generate plot with forecast data
        visualize_prediction(
            train_df=train,
            test_df=test,
            pred_df=forecast_df,
            title="Out of sample forecast",
            path_to_save_fig=os.path.join(output_dir, "figures"),
            file_name="historical_with_forecast_by_arima.png"
        )
        logging.info("data visualization is ready with forecasts and saved into output/figures directory")

        # generate file output
        generate_output(
            hist_df=full_data,
            forecast_df=forecast_df,
            path_to_save_op=os.path.join(output_dir, "data"),
            file_name="hist_2022_forecast_2030_arima.csv"
        )
        logging.info("forecasts with historical data are now combined and saved into output/data directory")


    except Exception as e:
        print(str(e))




if __name__ == "__main__":
    main()