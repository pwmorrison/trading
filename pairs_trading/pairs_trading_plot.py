import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from glob import glob
import pandas as pd
from collections import defaultdict
import os
import math
from datetime import datetime


def extract_pair(data_filename):
    stocks = os.path.splitext(os.path.basename(data_filename))[0].split('_')[2]
    pair = tuple(stocks.split('-'))
    return pair


def extract_datetime(data_filename):
    date_string = os.path.splitext(os.path.basename(data_filename))[0].split('_')[0]
    dt = datetime.strptime(date_string, '%Y-%m-%d')
    return dt


def get_latest_data(data_filename):
    # Get data from the bottom row.
    df = pd.read_csv(data_filename)
    data = df.iloc[-1][['date', 'close', 'mean', 'bollinger_top', 'bollinger_bottom']]
    return data


def plot_pair_data(filenames, ax):
    # Extract data from the data files.
    filenames.sort()
    data_list = [get_latest_data(f) for f in filenames]
    print(data_list)
    # dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in data_list]
    dates = [extract_datetime(f) for f in filenames]
    closes = [d['close'].item() for d in data_list]
    means = [d['mean'].item() for d in data_list]
    bollinger_tops = [d['bollinger_top'].item() for d in data_list]
    bollinger_bottom = [d['bollinger_bottom'].item() for d in data_list]
    print(dates)
    print(closes)

    # ax.scatter(dates, closes)
    ax.plot(dates, closes)
    ax.plot(dates, means)
    ax.plot(dates, bollinger_tops)
    ax.plot(dates, bollinger_bottom)


def plot_pair_trades(filenames, ax):
    pass


def main():
    trading_dir = Path(r'D:\data\pairs_trading\out')
    data_template = '*_data_*.csv'
    trades_template = '*_trades.csv'
    positions_template = '*_positions.csv'

    data_filenames = glob(str(trading_dir / data_template))
    print(data_filenames)

    # Group by pair.
    pair_data_filenames = defaultdict(list)
    for f in data_filenames:
        pair_data_filenames[extract_pair(f)].append(f)
    print(pair_data_filenames)

    # Plot each pair.
    num_pairs = len(pair_data_filenames)
    num_cols = 3
    num_rows = math.ceil(num_pairs / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols)
    pairs = list(pair_data_filenames.keys())
    pairs.sort()
    for pair, ax in zip(pairs, axes.flatten()):
        data_filenames = pair_data_filenames[pair]
        print(pair)
        print(data_filenames)
        plot_pair_data(data_filenames, ax)

        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

    fig.autofmt_xdate()
    plt.show()


if __name__ == '__main__':
    main()
