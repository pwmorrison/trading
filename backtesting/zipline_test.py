from zipline.api import order, record, symbol
from zipline import run_algorithm
import pandas as pd
import matplotlib.pyplot as plt


def initialize(context):
    pass


def handle_data(context, data):
    order(symbol('AAPL'), 10)
    record(AAPL=data.current(symbol('AAPL'), 'price'))


if __name__ == '__main__':
    perf = run_algorithm(
        start=pd.to_datetime('2016-01-01').tz_localize('US/Eastern'),
        end=pd.to_datetime('2018-01-01').tz_localize('US/Eastern'),
        initialize=initialize,
        handle_data=handle_data,
        capital_base=10000,
        bundle='quandl'
    )

    print(perf.head())

    # figsize(12, 12)


    ax1 = plt.subplot(211)
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('Portfolio Value')
    ax2 = plt.subplot(212, sharex=ax1)
    perf.AAPL.plot(ax=ax2)
    ax2.set_ylabel('AAPL Stock Price')

    plt.show()
