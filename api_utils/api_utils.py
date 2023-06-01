# -*- coding: utf-8 -*-

"""
This script provides functions to fetch data from World Bank API
author: Koushik Khan
"""

import json
import requests


def create_url(country_code, indicator, page_number, base_url="https://api.worldbank.org/v2"):
    # create full url
    full_url = f"{base_url}/country/{country_code}/indicator/{indicator}?format=json&page={page_number}"
    return full_url

def make_api_get_call(url):
    r = requests.get(url)
    r_dict = json.loads(r.text)
    return r_dict

def browse_pages(data_container):
    pass

def get_api_data_as_json(country_code="afg", indicator="NY.GDP.MKTP.CN", base_url="https://api.worldbank.org/v2", max_page=20):
    
    # initialize containers
    data_dict = []
    current_page = 1
    
    # create full url
    full_url = create_url(country_code=country_code, indicator=indicator, page_number=current_page)
    
    # make API call
    r_dict = make_api_get_call(url=full_url)

    # note: first call gives meta information, that is why 
    # we have started with `current_page = 1`

    n_pages = r_dict[0]["pages"]
    data_dict.append(r_dict[1])

    page_numbers = (i for i in range(2, max_page))
    for i in page_numbers:
        print(i)
        # update current_page
        current_page = i
        full_url = create_url(country_code="afg", indicator="NY.GDP.MKTP.CN", page_number=current_page)
        # print(f"test: {full_url}")

        # make API call
        r_dict = make_api_get_call(url=full_url)

        # extract and append data
        data_dict.append(r_dict[1])

        if i == n_pages:
            page_numbers.close()

    return data_dict


if __name__ == "__main__":
    # define tests
    pass