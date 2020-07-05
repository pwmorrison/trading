import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import glob
from pathlib import Path
import os
from datetime import datetime
import random


class RankingDataset(Dataset):

    def __init__(self, tickers_dir, fundamentals_filename, sectors, x_length, y_length, lookback, min_sequence_length, template='*.csv', transform=None,
                 start_date=None, end_date=None, fixed_start_date=False, datetime_format='%Y-%m-%d',
                 normalised_returns=False, max_n_files=None):
        self.x_length = x_length
        self.y_length = y_length
        self.series_length = x_length + y_length
        self.lookback = lookback
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        self.datetime_format = datetime_format
        self.fixed_start_date = fixed_start_date
        self.normalised_returns = normalised_returns

        self.fundamentals_df = pd.read_csv(fundamentals_filename)
        if sectors is None:
            sectors = self.fundamentals_df['sector'].unique()
        self.sectors = sectors
        print(f'Forming dataset from {len(self.sectors)} sectors: {self.sectors}.')

        # The tickers that are within the specified sectors.
        sectors_df = self.fundamentals_df[self.fundamentals_df['sector'].isin(sectors)]
        print(sectors_df.to_string())
        sector_tickers = sectors_df['ticker'].tolist()

        # Only keep tickers with length less than a minimum.
        ticker_files = glob.glob(str(Path(tickers_dir) / template))
        self.ticker_files = []
        print(f'Finding tickers with sufficient length, from {len(ticker_files)} files.')
        for ticker_file in ticker_files:

            ticker = os.path.splitext(os.path.basename(ticker_file))[0].split('_')[0]
            if ticker not in sector_tickers:
                continue

            if 0:
                if os.path.basename(ticker_file) != 'MS_prices.csv':
                    continue

            df = read_csv(ticker_file, self.datetime_format)

            # Identify the indices within the given date range.
            if start_date and end_date:
                valid_indices = ((df.index >= start_date) & (df.index <= end_date))
            elif start_date:
                valid_indices = df.index >= start_date
            elif end_date:
                valid_indices = df.index <= end_date
            else:
                assert False

            if sum(valid_indices) == 0:
                # No days in the given date range.
                continue

            # Get the indicies of the valid days in the date range.
            valid_indices = np.where(valid_indices)[0]

            if valid_indices[0] < lookback:
                # Not enough days before the start date to compute returns.
                continue

            if len(valid_indices) < self.series_length:
                # Not enough days between the start and end date.
                continue

            self.ticker_files.append(ticker_file)

        if max_n_files:
            ticker_files = ticker_files[:max_n_files]

        print(f'Found {len(self.ticker_files)} files: {self.ticker_files}')

    def __len__(self):
        return len(self.ticker_files)

    def get_random_series(self, start_date, available_filenames):
        """
        Gets a random series from a random ticker, starting at the given date.
        """
        while True:
            # Choose a random ticker filename.
            filename = random.choice(available_filenames)
            df = read_csv(filename, self.datetime_format)
            # Calculate a rolling z-score.
            df['returns'] = df['close'].pct_change()
            if self.normalised_returns:
                # Normalised returns.
                df['returns'] = (df['returns'] - df['returns'].rolling(self.lookback).mean()) / df['returns'].rolling(
                    self.lookback).std()
            df = df.dropna()
            df = df.loc[self.start_date:]
            df = df.iloc[:self.series_length]
            if df.shape[0] < self.series_length:
                # This series doesn't contain the required number of values.
                continue
            return df

    def read_ticker_df(self, filename):
        df = read_csv(filename, self.datetime_format)

        # Calculate a rolling z-score.
        df['returns'] = df['close'].pct_change()
        if self.normalised_returns:
            # Normalised returns.
            df['returns'] = (df['returns'] - df['returns'].rolling(self.lookback).mean()) / df['returns'].rolling(self.lookback).std()
        df = df.dropna()

        # Filter dates *after* calculating returns, so previous dates can be used in the lookback.
        if self.start_date:
            df = df.loc[self.start_date:]
        if self.end_date:
            df = df.loc[:self.end_date]

        return df

    def get_x_day_returns(self, close_series):
        """
        Calculates the returns from every element to the last element.
        """
        final_close = close_series[-1]
        x_day_returns = (final_close - close_series) / close_series
        return x_day_returns

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Loop until we find two random tickers with enough overlapping dates.
        while True:
            # Get a random filename.
            filename_1, filename_2 = random.sample(self.ticker_files, 2)
            df_1 = self.read_ticker_df(filename_1)
            df_2 = self.read_ticker_df(filename_2)

            df = pd.merge(df_1, df_2, how='inner', on='date')
            # df.to_csv('merged.csv')

            if df.shape[0] < self.series_length:
                continue

            returns_1 = np.array(df['returns_x'])
            returns_2 = np.array(df['returns_y'])

            close_1 = np.array(df['close_x'])
            close_2 = np.array(df['close_y'])

            if self.fixed_start_date:
                # Get a series at the start date.
                start = 0
            else:
                # Get a random sub-series.
                start = np.random.randint(0, returns_1.shape[0] - self.series_length)

            # x_length = self.series_length // 2
            # y_length = self.series_length - x_length
            start_y = start + self.x_length

            def extract_series(returns, start_index, length):
                series = returns[start_index: start_index + length]
                # Add extra feature dimension.
                series = series[..., np.newaxis]
                # Convert to torch tensor.
                series = torch.from_numpy(series)
                return series

            # Get the return series.
            series_1 = extract_series(returns_1, start, self.x_length)
            series_2 = extract_series(returns_2, start, self.x_length)

            # Get the x-day returns.
            close_series_1 = extract_series(close_1, start, self.x_length)
            close_series_2 = extract_series(close_2, start, self.x_length)
            x_day_returns_1 = self.get_x_day_returns(close_series_1)
            x_day_returns_2 = self.get_x_day_returns(close_series_2)

            # Use the close series to determine y.
            y_series_1 = extract_series(close_1, start_y, self.y_length)
            y_series_2 = extract_series(close_2, start_y, self.y_length)
            y_return_1 = (y_series_1[-1] - y_series_1[0]) / y_series_1[0]
            y_return_2 = (y_series_2[-1] - y_series_2[0]) / y_series_2[0]
            y = 1 if y_return_1 > y_return_2 else 0

            dates = df.index.values[start: start + self.series_length]
            dates = [str(d) for d in dates]

            sample = {
                'series_1': series_1,
                'series_2': series_2,
                'returns_1': x_day_returns_1,
                'returns_2': x_day_returns_2,
                'y': y,
                'filename_1': filename_1,
                'filename_2': filename_2,
                'dates': dates,
            }
            return sample


def read_csv(filename, datetime_format):
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'], format=datetime_format)
    df.set_index('date', inplace=True)
    return df


def create_ranking_dataset(tickers_dir, fundamentals_filename, sectors, x_length, y_length, lookback, min_sequence_length,
                          start_date=None, end_date=None, fixed_start_date=False,
                          normalised_returns=False, max_n_files=None):
    dataset = RankingDataset(tickers_dir, fundamentals_filename, sectors, x_length, y_length, lookback, min_sequence_length,
                            start_date=start_date, end_date=end_date, fixed_start_date=fixed_start_date,
                            normalised_returns=normalised_returns, max_n_files=max_n_files)
    return dataset


def test_ranking_dataset():
    tickers_dir = r'D:\projects\trading\mlbootcamp\tickers'
    fundamentals_filename = r'D:\snapshot_df.csv'
    # series_length = 200
    x_length = 20
    y_length = 5
    lookback = 200
    min_sequence_length = 2 * (x_length + y_length + lookback)
    max_n_files = 100
    start_date = datetime.strptime('2010/01/01', '%Y/%m/%d')
    sectors = ['Healthcare']

    dataset = create_ranking_dataset(tickers_dir, fundamentals_filename, sectors, x_length, y_length, lookback, min_sequence_length,
                                     start_date=start_date, max_n_files=max_n_files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for b, batch in enumerate(dataloader):
        series_1 = batch['series_1'].numpy()
        series_2 = batch['series_2'].numpy()
        returns_1 = batch['returns_1'].numpy()
        returns_2 = batch['returns_2'].numpy()
        filename_1 = batch['filename_1']
        filename_2 = batch['filename_2']
        dates = batch['dates']
        print(b, series_1.shape, filename_1, filename_2, dates)
        print('Returns series:')
        print(series_1[0, :20, 0])
        print(series_2[0, :20, 0])
        print('X-day returns:')
        print(returns_1[0, :20, 0])
        print(returns_2[0, :20, 0])

        break


def main():

    df = pd.read_csv(r'D:\projects\trading\mlbootcamp\tickers\AAPL_prices.csv')

    # Calculate a rolling z-score.
    lookback = 200
    df['returns'] = df['close'].pct_change()
    df['returns'] = (df['returns'] - df['returns'].rolling(lookback).mean()) / df['returns'].rolling(lookback).std()

    fig, axes = plt.subplots(1, 1, squeeze=False)
    ax = axes[0, 0]
    # ax.plot(df['date'], df['close'])


    # df['close'].plot()
    df['returns'].plot()
    # df['mean'].plot()



    plt.show()

    print(df.info())


if __name__ == '__main__':
    # test_ticker_dataset()
    test_ranking_dataset()
    # main()
