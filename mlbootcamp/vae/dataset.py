import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import glob
from pathlib import Path


class TickerDataset(Dataset):

    def __init__(self, root_dir, series_length, lookback, min_sequence_length, template='*.csv', transform=None,
                 start_date=None, end_date=None, fixed_start_date=False, datetime_format='%Y-%m-%d', max_n_files=None):
        self.series_length = series_length
        self.lookback = lookback
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        self.datetime_format = datetime_format
        self.fixed_start_date = fixed_start_date

        # Only keep tickers with length less than a minimum.
        ticker_files = glob.glob(str(Path(root_dir) / template))
        if max_n_files:
            ticker_files = ticker_files[:max_n_files]
        self.ticker_files = []
        print(f'Finding tickers with sufficient length, from {len(ticker_files)} files.')
        for ticker_file in ticker_files:
            df = self._read_csv(ticker_file)

            if self.start_date:
                df = df.loc[self.start_date:]
            if self.end_date:
                df = df.loc[:self.end_date]

            if df.shape[0] >= min_sequence_length:
                self.ticker_files.append(ticker_file)
        print(f'Found {len(self.ticker_files)} files.')

    def __len__(self):
        return len(self.ticker_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.ticker_files[idx]
        df = self._read_csv(filename)

        # Calculate a rolling z-score.
        df['returns'] = df['close'].pct_change()
        df['returns'] = (df['returns'] - df['returns'].rolling(self.lookback).mean()) / df['returns'].rolling(self.lookback).std()
        df = df.dropna()

        # Filter dates *after* calculating returns, so previous dates can be used in the lookback.
        if self.start_date:
            df = df.loc[self.start_date:]
        if self.end_date:
            df = df.loc[:self.end_date]

        returns = np.array(df['returns'])

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

    def _read_csv(self, filename):
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'], format=self.datetime_format)
        df.set_index('date', inplace=True)
        return df


def create_ticker_dataset(root_dir, series_length, lookback, min_sequence_length,
                          start_date=None, end_date=None, fixed_start_date=False, max_n_files=None):
    dataset = TickerDataset(root_dir, series_length, lookback, min_sequence_length,
                            start_date=start_date, end_date=end_date, fixed_start_date=fixed_start_date,
                            max_n_files=max_n_files)
    return dataset


def test_ticker_dataset():
    root_dir = r'D:\projects\trading\mlbootcamp\tickers'
    series_length = 200
    lookback = 200
    min_sequence_length = 2 * (series_length + lookback)
    max_n_files = 100

    dataset = create_ticker_dataset(root_dir, series_length, lookback, min_sequence_length, max_n_files=max_n_files)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for b, batch in enumerate(dataloader):
        series = batch['series']
        filename = batch['filename']
        dates = batch['dates']
        print(b, series.shape, filename, dates)


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
    test_ticker_dataset()
    # main()
