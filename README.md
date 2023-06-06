# Introduction

Hi there,

Welcome to the README of the repository [automated-forecasting-pipeline](https://github.com/koushikkhan/automated-forecasting-pipeline).

In this project, I have tried to focus on developing an automated pipeline for developing time series models based macro economic data provided by the World Bank.

It is true that time series analysis is something which is more of an activity of art than science. So, completely automating the modelling process may not be possible, but we can automate many of the surrounding things we do and reduce the manual effort as mush as possible.

In the subsequent sections, I will help you to understand the codebase and play with it. 

# Setting up the codebase

The code is written entirely in Python (3.10.10) and does need few specific libraries to work properly.

Once you are ready with Python 3.10.10, follow the command below to install the necessary packages on your system.

- clone the repository
  ```
  git clone https://github.com/koushikkhan/automated-forecasting-pipeline.git
  ```

- install the required libraries
  ```
  pip install -r requirements.txt
  ```

# Triggering the driver file

The driver (a.k.a main) file is there within the root of the repository as `main.py`, and this is the entry point for the application.

You can simply trigger this file by using the command `python main.py` (on Windows) or `python3 main.py` (on Linux/Mac).

# Navigating the repository

The entire application is divided into multiple modules to make it easily maintainable.

Find the descriptions below for all of them.

- `config/config.ini`: configuration file containing parameters to work with the [world bank api](https://datahelpdesk.worldbank.org/knowledgebase/articles/898581) as well as related to the time series modelling.

- `api_utils/api_utils.py`: a module, having the required functionalities to pull data from the world bank api and parse it accordingly.

- `pipeline/data_transformer.py`: a module, having couple of classes that are supposed to perform data transformation (like missing value imputation etc.), it can be extended further if needed.

- `model_utils/model_utils.py`: a module, having separate classes for developing models with ARIMA as well as [fbprophet](https://facebook.github.io/prophet/docs/quick_start.html#python-api) and some standalone utility functions.

- `output/data`: this directory contains the data files that hold actual values and the forecasted values.

- `output/figures`: as the name suggests, it contains figures (visualizations) related to forecasting.

- `output/model`: this directory contains binary files (**.pkl**) of the models that the application has generated. 

That's all, thanks.
