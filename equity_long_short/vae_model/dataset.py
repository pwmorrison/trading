import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
import os

import torch
from torch.utils.data import Dataset, DataLoader

"""
VAE PyTorch dataset.
"""

def read_ticker_csv(filename, datetime_format):
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'], format=datetime_format)
    df.set_index('date', inplace=True)
    return df


class TickerDataset(Dataset):
    """
    Provides returns series of single stocks.
    """

    def __init__(self, root_dir, series_length, lookback, min_sequence_length, template='*.csv', transform=None,
                 start_date=None, end_date=None, fixed_start_date=False, datetime_format='%Y-%m-%d',
                 normalised_returns=False, max_n_files=None):
        """
        lookback -- The lookback used to compute rolling z-score, when normalising returns.
        fixed_start_date -- If True, always returns the series at the start date of it's history. If False,
        returns random series within its history.
        """
        self.series_length = series_length
        self.lookback = lookback
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        self.datetime_format = datetime_format
        self.fixed_start_date = fixed_start_date
        self.normalised_returns = normalised_returns

        # Only keep tickers with length less than a minimum.
        ticker_files = glob(str(Path(root_dir) / template))
        if max_n_files:
            ticker_files = ticker_files[:max_n_files]
        self.ticker_files = []
        print(f'Finding tickers with sufficient length, from {len(ticker_files)} files.')
        for ticker_file in ticker_files:
            df = read_ticker_csv(ticker_file, self.datetime_format)

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

            if len(valid_indices) < series_length:
                # Not enough days between the start and end date.
                continue

            self.ticker_files.append(ticker_file)

        print(f'Found {len(self.ticker_files)} files of sufficient length.')

    def __len__(self):
        return len(self.ticker_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.ticker_files[idx]
        df = read_ticker_csv(filename, self.datetime_format)

        # Calculate a rolling z-score.
        df['returns'] = df['close'].pct_change()
        if self.normalised_returns:
            # Normalised returns.
            df['returns'] = (df['returns'] - df['returns'].rolling(self.lookback).mean()) / (
                df['returns'].rolling(self.lookback).std())

        if 0:
            # Return close prices.
            df['returns'] = df['close']

        df = df.dropna()

        # Filter dates *after* calculating returns, so previous dates can be used in the lookback.
        if self.start_date:
            df = df.loc[self.start_date:]
        if self.end_date:
            df = df.loc[:self.end_date]

        returns = np.array(df['returns'])

        # Extract a sub-series from this history.
        if self.fixed_start_date:
            # Get a series at the start date.
            start = 0
        else:
            # Get a random sub-series.
            start = np.random.randint(0, returns.shape[0] - self.series_length)
        series = returns[start: start + self.series_length]

        # Add extra feature dimension.
        series = series[..., np.newaxis]
        # Convert to torch tensor.
        series = torch.from_numpy(series)

        dates = df.index.values[start: start + self.series_length]
        dates = [str(d) for d in dates]

        sample = {
            'series': series,
            'filename': filename,
            'dates': dates,
        }

        return sample


def create_ticker_dataset(root_dir, series_length, lookback, min_sequence_length,
                          start_date=None, end_date=None, fixed_start_date=False,
                          normalised_returns=False, max_n_files=None):
    dataset = TickerDataset(root_dir, series_length, lookback, min_sequence_length,
                            start_date=start_date, end_date=end_date, fixed_start_date=fixed_start_date,
                            normalised_returns=normalised_returns, max_n_files=max_n_files)
    return dataset


def test_ticker_dataset():
    root_dir = r'E:\data\mlbootcamp\tickers'
    series_length = 200
    lookback = 20
    min_sequence_length = 2 * (series_length + lookback)
    max_n_files = 100
    start_date = "2010/01/01"
    end_date = "2018/01/01"

    dataset = create_ticker_dataset(root_dir, series_length, lookback, min_sequence_length,
        start_date=start_date, end_date=end_date, max_n_files=max_n_files)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for b, batch in enumerate(dataloader):
        # series is a tensor of shape [batch_size, series_length, 1]
        series = batch['series']
        # filename is a list of length batch_size
        filename = batch['filename']
        # dates is a list of length series_length. Each element is a list of length batch_size.
        dates = batch['dates']

        print(series.shape, len(filename), len(dates))
        print(len(dates[0]))


if __name__ == '__main__':
    test_ticker_dataset()
