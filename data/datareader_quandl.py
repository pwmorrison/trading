import pandas as pd
import datetime

import quandl
quandl.ApiConfig.api_key = "YOURAPIKEY"


def get_stock(stock_name, start_datetime, end_datetime, api='iex'):

    # The 'google' api doesn't seem to work any more.
    # iex
    # morningstar
    # stock = web.DataReader(stock_name, api, start_datetime, end_datetime)
    return None#stock


def main():
    print('Hello')

    stock_name = "FB"
    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime(2017, 1, 1)
    stock = get_stock(stock_name, start, end)

    # print(stock.head())


if __name__ == '__main__':
    main()
