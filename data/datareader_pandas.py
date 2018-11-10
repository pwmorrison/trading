import pandas as pd
import datetime

# Hack to import pandas-datareader, which doesn't seem to be updated to the latest pandas.
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web


def get_stock(stock_name, start_datetime, end_datetime, api='iex'):

    # The 'google' api doesn't seem to work any more.
    # iex
    # morningstar
    stock = web.DataReader(stock_name, api, start_datetime, end_datetime)
    return stock


def main():
    print('Hello')

    stock_name = "FB"
    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime(2017, 1, 1)
    stock = get_stock(stock_name, start, end)

    print(stock.head())


if __name__ == '__main__':
    main()
