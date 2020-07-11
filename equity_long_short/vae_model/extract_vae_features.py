import argparse
import json
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from glob import glob
import os
import subprocess
import pandas as pd
import scipy.stats as stats
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pyro

from model import VAE
from dataset import read_ticker_csv


def find_similar(vae, dataloader, cuda):
    # Find the latent space vector for every example in the test set.
    x_all = []
    z_all = []
    x_reconst_all = []
    filenames_all = []
    for batch in dataloader:
        # x, z, x_reconst = test_minibatch(dmm, test_batch, args, sample_z=True)
        x = batch['series']
        if cuda:
            x = x.cuda()
        x = x.float()
        x_reconst = vae.reconstruct_img(x)
        z_loc, z_scale, z = vae.encode_x(x)
        x = x.cpu().numpy()
        x_reconst = x_reconst.cpu().detach().numpy()
        z_loc = z_loc.cpu().detach().numpy()

        x_all.append(x)
        z_all.append(z_loc)
        x_reconst_all.append(x_reconst)
        filenames_all.extend(batch['filename'])
    x_all = np.concatenate(x_all, axis=0)
    z_all = np.concatenate(z_all, axis=0)
    x_reconst_all = np.concatenate(x_reconst_all, axis=0)


    # Get the closest latent to the query.
    n_neighbours = 5
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1 + n_neighbours, algorithm='ball_tree').fit(z_all)
    distances, indices = nbrs.kneighbors(z_all)

    query_indices = [0, 1]
    query_indices = np.random.randint(0, len(filenames_all), size=10)

    for query_index in query_indices:
        print(f'Query index {query_index}')
        # Skip the first closest index, since it is just the query index.
        closest_indices = indices[query_index][1:]

        # Plot the query and closest series.
        fig, axes = plt.subplots(1 + n_neighbours, 1, squeeze=False)
        ax = axes[0, 0]
        x_series = x_all[query_index, ...]
        ax.plot(range(x_series.shape[0]), x_series, c='r')
        ax.set_title(filenames_all[query_index])
        ax.grid()
        for i in range(n_neighbours):
            ax = axes[i + 1, 0]
            x_series = x_all[closest_indices[i], ...]
            ax.plot(range(x_series.shape[0]), x_series, c='b')
            ax.set_title(filenames_all[closest_indices[i]])
            ax.grid()
    plt.show()

    return


def select_next_period(z_encodings, filenames, n_pairs, fundamentals_df):
    """
    Selects the pairs to trade in the next period, using the most recent period.


    """
    # Get the closest latent to the query.
    n_neighbours = 5
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1 + n_neighbours, algorithm='ball_tree').fit(z_encodings)
    distances, indices = nbrs.kneighbors(z_encodings)

    # Get the smallest distances.
    # Remove the first column, which is the distance between each point and itself.
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    distances_shape = distances.shape
    # The sorted indices of the smallest distances in the distances array.
    sorted_indices = np.dstack(np.unravel_index(np.argsort(distances.ravel()), distances_shape))
    sorted_indices = sorted_indices[0]
    # print(distances)
    # print(sorted_indices)

    # Get the indices of the actual pairs in the input array.
    # pairs_indices = sorted_indices[:n_pairs]
    selected_pairs = []
    for i in range(sorted_indices.shape[0]):
        # print(pairs_indices[i])
        ind_1 = sorted_indices[i, 0]
        ind_2 = indices[sorted_indices[i, 0], sorted_indices[i, 1]]
        pair = [ind_1, ind_2]
        # pairs_indices[i, 1] = indices[pairs_indices[i, 0], pairs_indices[i, 1]]
        # print(pairs_indices[i])

        # Check if the same pair (in different order) is already in the selected pairs.
        if [ind_2, ind_1] in selected_pairs:
            continue

        # Check if the pairs have the same sector and industry.
        ticker_1 = get_ticker_from_filename(filenames[ind_1])
        ticker_2 = get_ticker_from_filename(filenames[ind_2])
        sector_1, industry_1 = get_sector_and_industry(fundamentals_df, ticker_1)
        sector_2, industry_2 = get_sector_and_industry(fundamentals_df, ticker_2)
        if sector_1 != sector_2 or industry_1 != industry_2:
            continue

        selected_pairs.append(pair)

        if len(selected_pairs) == n_pairs:
            break
    selected_pairs = np.array(selected_pairs)

    return selected_pairs


def get_ticker_from_filename(filename):
    ticker = os.path.splitext(os.path.basename(filename))[0].split('_')[0]
    return ticker


def get_sector_and_industry(fundamentals_df, ticker):
    sector = fundamentals_df.loc[ticker]['sector']
    industry = fundamentals_df.loc[ticker]['industry']
    return sector, industry


def get_vae_embeddings(x_all, filenames_all, vae, n_pairs, fundamentals_df, cuda):
    z_all = {}
    x_reconst_all = {}
    for ticker, x in x_all.items():
        # Get the VAE z encoding of this series.

        # Add extra feature dimension.
        series = x[..., np.newaxis]
        # Convert to torch tensor.
        x = torch.from_numpy(series)
        if cuda:
            x = x.cuda()
        x = x.float()
        z_loc, z_scale, z = vae.encode_x(x)
        z_loc = z_loc.cpu().detach().numpy()

        x_reconst = vae.reconstruct_img(x)
        x_reconst = x_reconst.cpu().detach().numpy()

        z_all[ticker] = z_loc[0]
        x_reconst_all[ticker] = x_reconst[0]
        # z_all.append(z_loc)
        # x_reconst_all.append(x_reconst[0])

    # z_all = np.concatenate(z_all, axis=0)

    return z_all


def plot_hist(vals, hist=True):
    fig, axes = plt.subplots(1, 1, squeeze=False)
    ax = axes[0, 0]
    if hist:
        ax.hist(vals)
    else:
        y = [0] * len(vals)
        ax.plot(vals, y, 'bo', alpha=0.5)
    mu = np.mean(vals)
    sd = np.std(vals)
    x = np.linspace(min(vals), max(vals), 100)
    ax2 = ax.twinx()
    ax2.plot(x, stats.norm.pdf(x, mu, sd), c='r')
    ax2.axvline(mu, c='r')
    ax.grid()
    return fig, axes


def plot_backtest_results(returns=None, std_devs=None, sharpes=None, hist=True):
    fig, axes = plt.subplots(1, 3, squeeze=False, figsize=(20, 10))

    for i, (vals, title) in enumerate(zip([returns, std_devs, sharpes], ['Returns', 'SDs', 'Sharpes'])):
        if vals is None or len(vals) == 0:
            continue

        ax = axes[0, i]
        if hist:
            ax.hist(vals)
        else:
            y = np.random.uniform(-0.1, 0.1, len(vals))
            ax.plot(vals, y, 'bo', alpha=0.5)
        mu = np.mean(vals)
        sd = np.std(vals)
        x = np.linspace(min(vals), max(vals), 100)
        ax2 = ax.twinx()
        ax2.plot(x, stats.norm.pdf(x, mu, sd), c='r')
        ax2.axvline(mu, c='r')
        ax.grid()
        ax.set_title(title)
        ax.set_ylim([-1, 1])
        ax.axvline(0, c='black')

    return fig, axes


def get_returns_series(ticker_files, current_date, n_days, rolling_lookback, datetime_format):
    """
    Calculates and returns the VAE embeddings at the given date.
    """
    x_all = {}
    x_mean_all = {}
    x_std_all = {}
    filenames_all = {}
    # tickers_all = []
    lookback_start_dates = {}
    for file in ticker_files:
        df = read_csv(file, datetime_format)

        # Get the lookback period ending at the current date.
        df = df.loc[:current_date]
        df = df.iloc[-(rolling_lookback + n_days):]

        # Calculate normalised returns.
        df['returns'] = df['close'].pct_change()
        df['mean'] = df['returns'].rolling(rolling_lookback).mean()
        df['std'] = df['returns'].rolling(rolling_lookback).std()
        df['returns'] = (df['returns'] - df['mean']) / df['std']
        df = df.dropna()

        # We should be left with the number of lookback days.
        if df.shape[0] != n_days:
            continue

        x = np.array(df['returns'])
        x_mean = np.array(df['mean'])
        x_std = np.array(df['std'])

        ticker = get_ticker_from_filename(file)
        x_all[ticker] = x
        x_mean_all[ticker] = x_mean
        x_std_all[ticker] = x_std
        filenames_all[ticker] = file
        lookback_start_dates[ticker] = df.index[0]
        # x_all.append(x)
        # x_mean_all.append(x_mean)
        # x_std_all.append(x_std)
        # filenames_all.append(file)
        # # tickers_all.append(get_ticker_from_filename(file))
        # lookback_start_dates.append(df.index[0])

    return x_all, x_mean_all, x_std_all, filenames_all, lookback_start_dates


def extract_vae_distances(start_date, end_date,
                     n_lookback_days, n_backtest_days, n_trade_days,
                     n_pairs_vae, n_pairs_backtest,
                     returns_lookback, vae, ticker_files, fundamentals_file, out_dir,
                     r_script_exe, r_backtest_script, r_trade_script, backtest_sd,
                     backtest_returns_file, backtest_plot_file, trade_returns_file, trade_plot_file,
                     datetime_format='%Y-%m-%d', cuda=False):
    """
    Runs walk forward starting at the given date, for some periods.
    """
    current_date = start_date

    # Read the fundamentals file, so we have access to sector and industry.
    fundamentals_df = pd.read_csv(fundamentals_file)
    df_sector_industry = fundamentals_df[['ticker', 'sector', 'industry']]
    df_sector_industry.set_index('ticker')

    # A file to record backtest data and outcomes, for training a predictor.
    backtest_results_file = open(Path(out_dir) / 'backtest_results.csv', 'w')

    distances_all_filename = Path(out_dir) / f'vae_distances_6periods_all.csv'
    distances_all_file = open(distances_all_filename, 'w')

    vae_offsets = [0]#, 30, 60, 120, 150, 180]
    vae_offsets = [0, 60, 120, 180, 240]

    header_row = f'stock1,stock2,year'
    for offset in vae_offsets:
        header_row += f',l1_offset_{offset}'
    header_row += ',l1_average'
    for offset in vae_offsets:
        header_row += f',l2_offset_{offset}'
    header_row += ',l2_average\n'
    distances_all_file.write(header_row)

    # Loop over walk forward periods.
    trade_returns = []
    trade_sds = []
    trade_sharpes = []
    # years = [2015, 2016, 2017, 2018, 2019]
    years = list(range(2020, 2020+1))
    for year in years:
    # while True:
        current_date = datetime.strptime(f'{year}/01/01', '%Y/%m/%d')
        print(f'Extracting VAE distances at date {current_date}')

        current_out_dir = Path(out_dir) / current_date.strftime("%Y-%m-%d")
        current_out_dir.mkdir(exist_ok=True)

        l1_distances = {}
        l2_distances = {}
        for vae_offset in vae_offsets:
            vae_date = current_date - timedelta(int(vae_offset * 7. / 5))
            print(vae_date)

            # Get the series for each ticker file.
            x_all, x_mean_all, x_std_all, filenames_all, lookback_start_dates = get_returns_series(
                ticker_files, vae_date, n_lookback_days, returns_lookback, datetime_format
            )

            # Get the VAE z-latents.
            z_all = get_vae_embeddings(x_all, filenames_all, vae, n_pairs_vae, fundamentals_df, cuda)

            # Compute distances.
            # distances = {}
            tickers = list(z_all.keys())
            all_z_latents = []
            for pair_1, ticker_1 in enumerate(tickers):
                z_1 = z_all[ticker_1]
                for pair_2, ticker_2 in enumerate(tickers[pair_1 + 1:]):
                    pair = (ticker_1, ticker_2)
                    if pair not in l2_distances:
                        l1_distances[pair] = []
                        l2_distances[pair] = []
                    z_2 = z_all[ticker_2]
                    l1_distance = np.sum(np.abs(z_1 - z_2))
                    l2_distance = np.sqrt(np.sum(np.square(z_1 - z_2)))
                    # distances[pair] = l2_distance
                    l1_distances[pair].append(l1_distance)
                    l2_distances[pair].append(l2_distance)
                all_z_latents.append(z_1[np.newaxis, ...])

            # Add the distances at this offset to all distances.
            # for pair, distance in distances.items():
            #     if pair not in distances_all:
            #         distances_all[pair] = []
            #     distances_all[pair].append(distance)

        # Write distances to file.
        distances_filename = Path(out_dir) / f'vae_distances_6periods_{current_date.strftime("%Y%m%d")}.csv'
        distances_file = open(distances_filename, 'w')
        distances_file.write(header_row)
        for (ticker_1, ticker_2) in l2_distances.keys():
            l1_distance_offsets = l1_distances[(ticker_1, ticker_2)]
            l2_distance_offsets = l2_distances[(ticker_1, ticker_2)]
            if len(l2_distance_offsets) != len(vae_offsets):
                continue
            row = f'{ticker_1},{ticker_2},{year}'
            for l1_distance in l1_distance_offsets:
                row += f',{l1_distance}'
            row += f',{np.mean(l1_distance)}'
            for l2_distance in l2_distance_offsets:
                row += f',{l2_distance}'
            row += f',{np.mean(l2_distance)}\n'
            distances_all_file.write(row)
            distances_file.write(row)
        distances_file.close()
        distances_all_file.flush()

        all_latents = np.concatenate(all_z_latents, axis=0)

        # Run t-SNE with 2 output dimensions.
        from sklearn.manifold import TSNE
        model_tsne = TSNE(n_components=2, random_state=0)
        # z_states = all_latents.detach().cpu().numpy()
        z_states = all_latents
        z_embed = model_tsne.fit_transform(z_states)
        # Write out to a file.
        tsne_2d_filename = Path(out_dir) / f'vae_tsne2d_{current_date.strftime("%Y%m%d")}.csv'
        tsne_2d_file = open(tsne_2d_filename, 'w')
        tsne_2d_file.write('ticker,x,y\n')
        for pair_1, ticker_1 in enumerate(tickers):#range(n_tickers):
            tsne_embed = z_embed[pair_1]
            tsne_2d_file.write(f'{ticker_1}')
            for i in range(tsne_embed.shape[0]):
                tsne_2d_file.write(f',{tsne_embed[i]}')
            tsne_2d_file.write(f'\n')
        tsne_2d_file.close()
        # Plot t-SNE embedding.
        fig = plt.figure()
        plt.scatter(z_embed[:, 0], z_embed[:, 1], s=10)
        fig.savefig(Path(out_dir) / f'tsne_test.png')
        plt.close(fig)

        # 3D t-SNE.
        model_tsne = TSNE(n_components=3, random_state=0)
        z_states = all_latents
        z_embed = model_tsne.fit_transform(z_states)
        tsne_3d_filename = Path(out_dir) / f'vae_tsne3d_{current_date.strftime("%Y%m%d")}.csv'
        tsne_3d_file = open(tsne_3d_filename, 'w')
        tsne_3d_file.write('ticker,x,y,z\n')
        for pair_1, ticker_1 in enumerate(tickers):#range(n_tickers):
            tsne_embed = z_embed[pair_1]
            tsne_3d_file.write(f'{ticker_1}')
            for i in range(tsne_embed.shape[0]):
                tsne_3d_file.write(f',{tsne_embed[i]}')
            tsne_3d_file.write(f'\n')
        tsne_3d_file.close()


        # Read the stored t-SNE coordinates and add some fundamental data.
        df = pd.read_csv(tsne_3d_filename)
        print(df.head())
        print(df_sector_industry.head())
        df = df.set_index('ticker').join(df_sector_industry.set_index('ticker'))
        print(df.head())
        df.to_csv(Path(out_dir) / f'vae_tsne3d_fund_{current_date.strftime("%Y%m%d")}.csv')


        # break

        # vae_pairs_filename = current_out_dir / 'vae_pairs.csv'
        # vae_pairs_file = open(vae_pairs_filename, 'wb')
        # for pair in pairs_indices_vae:
        #     vae_pairs_file.write(f'{tickers_all[pair[0]]},{tickers_all[pair[1]]}\n')
        # vae_pairs_file.close()

        # print(f'Selected {len(pairs_indices_vae)} pairs using VAE.')


        # # Write the selected pairs out to a file.
        # vae_pairs_filename = current_out_dir / 'vae_pairs.csv'
        # vae_pairs_file = open(vae_pairs_filename, 'wb')
        # for pair in pairs_indices_vae:
        #     vae_pairs_file.write(f'{tickers_all[pair[0]]},{tickers_all[pair[1]]}\n')
        # vae_pairs_file.close()

        trade_end_date = current_date + timedelta(int(n_trade_days * 7. / 5))  # Assume 5 trade days per week.

        # Move forward to the next period
        current_date = trade_end_date

        if current_date > end_date:
            break

    backtest_results_file.close()
    distances_all_file.close()

    print('Finished walk forward.')



def form_start_dates(start_date, end_date, period):
    """
    Determines the *approximate* start dates periods between the given dates.
    The dates are approximate because they may not coincide with trading days, and may not be spaced exactly
    series_length days apart.
    """
    dates = []
    period_td = timedelta(int(period * 7. / 5))
    period_start_date = start_date
    if 0:
        # Ensure each period is within the given range.
        period_end_date = period_start_date + period_td
        while period_end_date < end_date:
            dates.append(period_start_date)
            period_start_date += period_td
            period_end_date = period_start_date + period_td
    else:
        # Just ensure the start date of each period is within the given range.
        while period_start_date < end_date:
            dates.append(period_start_date)
            period_start_date += period_td
    return dates


def get_series_ending_at_date(ticker_file, end_date, series_length, returns_lookback, normalised_returns,
    datetime_format='%Y-%m-%d'):
    """
    """
    df = read_ticker_csv(ticker_file, datetime_format)

    # Only keep data up to the given end date.
    # TODO: Filter the start date too, to avoid redundant calculations.
    df = df[df.index < end_date]

    # Calculate a rolling returns.
    df['returns'] = df['close'].pct_change()
    if normalised_returns:
        # Normalised returns.
        df['returns'] = (df['returns'] - df['returns'].rolling(self.lookback).mean()) / (
            df['returns'].rolling(self.lookback).std())
    df = df.dropna()

    # Get the final series_length elements of the df.
    df = df.iloc[-series_length:]

    return df

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



def extract_vae_features(start_date, end_date, series_length, returns_lookback, vae, ticker_files,
    fundamentals_df, out_path):
    """
    Extracts the VAE features of each ticker at repeating consecutive periods within the given date range.
    """
    # Get the start dates of the periods we will extract features for.
    start_dates = form_start_dates(start_date, end_date, series_length)
    print(start_dates)

    # Get the returns series for the period ending at each start date.
    for current_date in start_dates:
        print(current_date)
        for ticker_file in ticker_files:
            print(ticker_file)
            input_series = get_series_ending_at_date(ticker_file, current_date, series_length, returns_lookback,
                normalised_returns=False)
            print(input_series)
            return

def main():
    parser = argparse.ArgumentParser(description='Train VAE.')
    parser.add_argument('-c', '--config', default='extract_vae_features_config.json', help='Config file.')
    args = parser.parse_args()
    print(args)
    c = json.load(open(args.config))
    print(c)

    in_path = Path(c['in_dir'])
    out_path = Path(c['out_dir'])

    start_date = datetime.strptime(c['start_date'], '%Y/%m/%d')
    end_date = datetime.strptime(c['end_date'], '%Y/%m/%d')

    vae = None

    max_n_files = 100

    # TODO: Create a function in dataset.py that will give us the ticker files that cover the given date range.
    # Take most of the code from the Dataset's constructor.
    ticker_files = glob(str(in_path / '*.csv'))
    if max_n_files:
        ticker_files = ticker_files[:max_n_files]
    print(ticker_files)

    fundamentals_df = pd.read_csv(c['fundamentals_file'])

    vae_features = extract_vae_features(start_date, end_date, c['vae_series_length'], c['returns_lookback'], vae,
        ticker_files, fundamentals_df, out_path)

def main2():
    parser = argparse.ArgumentParser(description='Train VAE.')
    parser.add_argument('-c', '--config', help='Config file.')
    args = parser.parse_args()
    print(args)
    c = json.load(open(args.config))
    print(c)

    # TODO: Move to config file.
    lookback = 50  # 160
    input_dim = 1
    returns_lookback = 20

    start_date = datetime.strptime(c['start_date'], '%Y/%m/%d') if c['start_date'] else None
    end_date = datetime.strptime(c['end_date'], '%Y/%m/%d') if c['end_date'] else None
    max_n_files = None

    out_path = Path(c['out_dir'])
    out_path.mkdir(exist_ok=True)

    vae = VAE(c['vae_series_length'], z_dim=c['z_dim'], use_cuda=c['cuda'])

    if c['checkpoint_load']:
        checkpoint = torch.load(c['checkpoint_load'])
        vae.load_state_dict(checkpoint['model_state_dict'])

    ticker_files = glob(str(Path(c['in_dir']) / '*.csv'))
    if 0:
        ticker_files = ticker_files[:100]
    print(f'Found {len(ticker_files)} ticker files.')

    extract_vae_distances(
        start_date=start_date,
        end_date=end_date,
        n_lookback_days=c['n_lookback_days'],
        n_backtest_days=c['n_backtest_days'],
        n_trade_days=c['n_trade_days'],
        n_pairs_vae=c['n_pairs_vae'],
        n_pairs_backtest=c['n_pairs_backtest'],
        vae=vae,
        returns_lookback=returns_lookback,
        ticker_files=ticker_files,
        fundamentals_file=c['fundamentals_file'],
        out_dir=c['out_dir'],
        r_script_exe=c['r_script_exe'],
        r_backtest_script=c['r_backtest_script'],
        r_trade_script=c['r_trade_script'],
        backtest_sd=c['backtest_sd'],
        backtest_returns_file=c['backtest_returns_file'],
        backtest_plot_file=c['backtest_plot_file'],
        trade_returns_file=c['trade_returns_file'],
        trade_plot_file=c['trade_plot_file'],
        cuda=c['cuda'])


if __name__ == '__main__':
    main()
