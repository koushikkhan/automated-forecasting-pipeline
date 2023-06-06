# -*- coding: utf-8 -*-

"""
This script provides functions to fetch data from World Bank API
author: Koushik Khan
"""

import json
import requests
import pandas as pd

def create_url(country_code, indicator, page_number, base_url):
    # create full url
    full_url = f"{base_url}/country/{country_code}/indicator/{indicator}?format=json&page={page_number}"
    return full_url


def make_api_get_call(url):
    r = requests.get(url)
    r_dict = json.loads(r.text)
    return r_dict


def get_api_data_as_json(country_code, indicator, base_url, max_page):
    
    # initialize containers
    data_dict = []
    current_page = 1
    
    # create full url
    full_url = create_url(country_code=country_code, indicator=indicator, page_number=current_page, base_url=base_url)
    
    # make API call
    r_dict = make_api_get_call(url=full_url)

    # note: first call gives meta information, that is why 
    # we have started with `current_page = 1`

    n_pages = r_dict[0]["pages"]
    data_dict.append(r_dict[1])

    page_numbers = (i for i in range(2, max_page))
    for i in page_numbers:
        # print(i)
        # update current_page
        current_page = i
        full_url = create_url(country_code=country_code, indicator=indicator, page_number=current_page, base_url=base_url)
        # print(f"test: {full_url}")

        # make API call
        r_dict = make_api_get_call(url=full_url)

        # extract and append data
        data_dict.append(r_dict[1])

        if i == n_pages:
            page_numbers.close()

    return data_dict


def parse_api_output_for_tsa(api_output, select_cols=['date', 'value'], sort_by_date=True, set_date_idx=False):
    if isinstance(api_output, list):
        # convert each data dictionary to dataframe
        df_list = [pd.DataFrame(api_output[i]) for i in range(len(api_output))]
    elif isinstance(api_output, dict):
        # most probably there is only one page in the API
        df_list = pd.DataFrame(api_output)
    else:
        raise(ValueError("api_output seems to be ambiguous"))
    
    # combine multiple df's (if required)
    df_combined = pd.concat(df_list, axis=0)

    # select columns
    df_combined_sel_cols = df_combined[select_cols]

    # parse date col
    df_combined_sel_cols["date"] = pd.to_datetime(df_combined_sel_cols["date"])

    if sort_by_date:
        df_combined_sel_cols.sort_values('date', inplace=True)

    # set 'date' as index
    if set_date_idx:
        df_combined_sel_cols.set_index('date', inplace=True)

    return df_combined_sel_cols


if __name__ == "__main__":
    # define tests
    pass