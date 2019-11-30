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
import torch
import pyro

from dataset import create_ticker_dataset, read_csv
from model import VAE


def select_next_period(z_encodings_dict, n_pairs, fundamentals_df):
    """
    Selects the pairs to trade in the next period, using the most recent period.
    """
    tickers = list(z_encodings_dict.keys())
    z_encodings = np.array(list(z_encodings_dict.values()))

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

    # Get the indices of the actual pairs in the input array.
    # pairs_indices = sorted_indices[:n_pairs]
    selected_pairs = []
    selected_distances = {}
    for i in range(sorted_indices.shape[0]):
        # print(pairs_indices[i])
        ind_1 = sorted_indices[i, 0]
        ind_2 = indices[sorted_indices[i, 0], sorted_indices[i, 1]]

        # pair = [tickers[ind_1], tickers[ind_2]]
        ticker_1 = tickers[ind_1]
        ticker_2 = tickers[ind_2]

        # Check if the same pair (in different order) is already in the selected pairs.
        if [ticker_2, ticker_1] in selected_pairs:
            continue

        # Check if the pairs have the same sector and industry.
        sector_1, industry_1 = get_sector_and_industry(fundamentals_df, ticker_1)
        sector_2, industry_2 = get_sector_and_industry(fundamentals_df, ticker_2)
        if sector_1 != sector_2 or industry_1 != industry_2:
            continue

        pair = [ticker_1, ticker_2]
        selected_pairs.append(pair)

        distance = distances[sorted_indices[i, 0], sorted_indices[i, 1]]
        selected_distances[tuple(pair)] = distance

        if len(selected_pairs) == n_pairs:
            break

    selected_pairs = np.array(selected_pairs)

    return selected_pairs, selected_distances


def get_ticker_from_filename(filename):
    ticker = os.path.splitext(os.path.basename(filename))[0].split('_')[0]
    return ticker


def get_sector_and_industry(fundamentals_df, ticker):
    sector = fundamentals_df.loc[ticker]['sector']
    industry = fundamentals_df.loc[ticker]['industry']
    return sector, industry


def get_vae_embeddings(x_all, vae, cuda):
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

    return z_all


def select_pairs_vae(x_all, vae, n_pairs, fundamentals_df, cuda):
    # Get the VAE z-latents.
    z_all = get_vae_embeddings(x_all, vae, cuda)

    # Use the z-encodings to select pairs to trade.
    pairs_indices, pair_distances = select_next_period(z_all, n_pairs, fundamentals_df)

    return pairs_indices, pair_distances


def select_pairs(backtest_results, pairs, n_pairs):

    # Assemble the sharpe ratios in order.
    backtest_sharpes = []
    for pair, (backtest_return, backtest_stddev, backtest_sharpe) in backtest_results.items():
        backtest_sharpes.append(backtest_sharpe)

    # Get the indices of the sorted backtest results.
    sorted_indices = np.flip(np.argsort(backtest_sharpes))
    sorted_indices = sorted_indices[:n_pairs]

    # Get the best backtest pairs.
    selected_pairs_indices = []
    for i in sorted_indices:
        selected_pairs_indices.append(pairs[i])
    selected_pairs_indices = np.array(selected_pairs_indices)

    return selected_pairs_indices


def backtest_pairs_bulk(pairs, start_date, end_date, out_dir, r_script_exe, r_backtest_script, backtest_sd):

    # Write the selected pairs out to a file.
    vae_pairs_filename = out_dir / 'vae_pairs.csv'
    vae_pairs_file = open(vae_pairs_filename, 'w')
    backtest_returns_files = []
    for pair in pairs:
        # ticker_1 = tickers_all[pair[0]]
        # ticker_2 = tickers_all[pair[1]]
        ticker_1 = pair[0]
        ticker_2 = pair[1]
        returns_filename = str(out_dir / f'backtest_returns_{ticker_1}-{ticker_2}.csv').replace(os.sep, os.altsep)
        plot_filename = str(out_dir / f'backtest_plot_{ticker_1}-{ticker_2}.png').replace(os.sep, os.altsep)
        vae_pairs_file.write(f'{ticker_1},{ticker_2},{returns_filename},{plot_filename}\n')
        backtest_returns_files.append(returns_filename)
    vae_pairs_file.close()

    # Run the backtests in one call to the R script.
    backtest_command = [
        r_script_exe,
        r_backtest_script,
        str(vae_pairs_filename),
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        str(backtest_sd)]
    print(f'Running selection backtest with command: {backtest_command}')
    subprocess.call(backtest_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Get the returns results.
    backtest_results = {}
    for i, backtest_returns_file in enumerate(backtest_returns_files):
        backtest_results_df = pd.read_csv(backtest_returns_file)
        backtest_return = backtest_results_df.iloc[0][1]
        backtest_stddev = backtest_results_df.iloc[1][1]
        backtest_sharpe = backtest_results_df.iloc[2][1]

        pair = pairs[i]
        backtest_results[tuple(pair)] = [backtest_return, backtest_stddev, backtest_sharpe]

    return backtest_results


def trade_pairs_bulk(pairs, start_date, end_date, out_dir, r_script_exe, r_script, backtest_sd):

    # Write the selected pairs out to a file.
    pairs_filename = out_dir / 'vae_pairs_trade.csv'
    pairs_file = open(pairs_filename, 'w')
    trade_returns_files = []
    for pair in pairs:
        ticker_1, ticker_2 = pair[0], pair[1]
        returns_filename = str(out_dir / f'trade_returns_{ticker_1}-{ticker_2}.csv').replace(os.sep, os.altsep)
        plot_filename = str(out_dir / f'trade_plot_{ticker_1}-{ticker_2}.png').replace(os.sep, os.altsep)
        pairs_file.write(f'{ticker_1},{ticker_2},{returns_filename},{plot_filename}\n')
        trade_returns_files.append(returns_filename)
    pairs_file.close()

    # Run the trades in one call to the R script.
    command = [
        r_script_exe,
        r_script,
        str(pairs_filename),
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        str(backtest_sd)]
    print(f'Running trade backtest with command: {command}')
    subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Get the returns results.
    backtest_results = {}
    for i, backtest_returns_file in enumerate(trade_returns_files):
        backtest_results_df = pd.read_csv(backtest_returns_file)
        backtest_return = backtest_results_df.iloc[0][1]
        backtest_stddev = backtest_results_df.iloc[1][1]
        backtest_sharpe = backtest_results_df.iloc[2][1]

        pair = pairs[i]
        backtest_results[tuple(pair)] = [backtest_return, backtest_stddev, backtest_sharpe]

    return backtest_results


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


def plot_backtest_results(backtest_results, hist=True):

    returns = [backtest_results[pair][0] for pair in backtest_results if not np.isnan(backtest_results[pair][0])]
    std_devs = [backtest_results[pair][1] for pair in backtest_results if not np.isnan(backtest_results[pair][1])]
    sharpes = [backtest_results[pair][2] for pair in backtest_results if not np.isnan(backtest_results[pair][2])]

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
        if 0:
            # Plot the average and normal curve.
            sd = np.std(vals)
            x = np.linspace(min(vals), max(vals), 100)
            ax2 = ax.twinx()
            ax2.plot(x, stats.norm.pdf(x, mu, sd), c='r')
            ax2.axvline(mu, c='r')
        else:
            # Just plot the average.
            ax.axvline(mu, c='r')
        ax.grid()
        ax.set_title(title)
        ax.set_ylim([-0.2, 0.2])
        ax.axvline(0, c='black')

        ax.set_yticklabels([])

    return fig, axes


def get_returns_series(ticker_files, current_date, n_days, rolling_lookback, datetime_format):
    """
    Calculates and returns the VAE embeddings at the given date.
    """
    x_all = {}
    x_mean_all = {}
    x_std_all = {}
    filenames_all = {}
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

    return x_all, x_mean_all, x_std_all, filenames_all, lookback_start_dates


def plot_vae_selections(pairs_vae, pairs_distances_vae, x_all, x_mean_all, x_std_all, fundamentals_df, out_dir):

    for pair_ind, pair in enumerate(pairs_vae):
        ticker_1 = pair[0]
        ticker_2 = pair[1]
        distance = pairs_distances_vae[tuple(pair)]
        sector_1, industry_1 = get_sector_and_industry(fundamentals_df, ticker_1)

        # Get the average mean and std. dev. of each sequence, to see if it says anything interesting.
        ticker_1_mean = np.mean(x_mean_all[ticker_1])
        ticker_2_mean = np.mean(x_mean_all[ticker_2])
        ticker_1_std = np.mean(x_std_all[ticker_1])
        ticker_2_std = np.mean(x_std_all[ticker_2])

        fig, axes = plt.subplots(1, 1, squeeze=False)
        plt.subplots_adjust(top=0.7)
        # fig.suptitle(f'{ticker_1} - {ticker_2}: {sector_1} ({industry_1})\n')
        ax = axes[0, 0]
        ax.plot(x_all[pair[0]], label=ticker_1)
        ax.plot(x_all[pair[1]], label=ticker_2)
        ax.legend(loc='upper left')
        ax.set_title(f'{ticker_1} - {ticker_2}: {sector_1} ({industry_1})\n'
                     f'{ticker_1} returns mean {ticker_1_mean:.4f}, std. dev. {ticker_1_std:.4f}\n'
                     f'{ticker_2} returns mean {ticker_2_mean:.4f}, std. dev. {ticker_2_std:.4f}\n'
                     f'VAE distance {distance:.4f}'
                     , pad=10)

        plt.savefig(out_dir / f'vae_selection_{ticker_1}-{ticker_2}.png')
        plt.close(fig)


def output_spreads(pairs_vae, filenames_all, current_date, n_backtest_days, returns_lookback, backtest_results, trade_results, backtest_results_file, out_dir, datetime_format):
    # Plot the rolling mean of the spread, for each pair that we backtested.
    # This is for generating features and targets for a supervised "profitability" predictor.
    # This requires the traded pairs to be the same as the backtested pairs (in config file).
    for pair_ind, pair in enumerate(pairs_vae):
        ticker_1, ticker_2 = pair[0], pair[1]
        filename_1 = filenames_all[ticker_1]
        filename_2 = filenames_all[ticker_2]

        df_1 = read_csv(filename_1, datetime_format)
        df_1 = df_1.loc[:current_date]
        df_1 = df_1.iloc[-(returns_lookback + n_backtest_days):]

        df_2 = read_csv(filename_2, datetime_format)
        df_2 = df_2.loc[:current_date]
        df_2 = df_2.iloc[-(returns_lookback + n_backtest_days):]

        # The spread between the two tickers.
        df_1['spread'] = df_1['close'] / df_2['close']

        # The rolling mean and std dev. of the spread.
        df_1['spread_mean'] = df_1['spread'].rolling(20).mean()
        df_1['spread_std'] = df_1['spread'].rolling(20).std()

        # Normalise the mean and std. dev.
        df_1['spread_norm'] = (df_1['spread'] - df_1['spread'].mean()) / df_1['spread'].std()
        df_1['spread_norm_mean'] = df_1['spread_norm'].rolling(20).mean()
        df_1['spread_norm_std'] = df_1['spread_norm'].rolling(20).std()
        # df_1['spread_mean_norm'] = (df_1['spread_mean'] - df_1['spread_mean'].mean()) / df_1['spread_mean'].std()
        # df_1['spread_std_norm'] = (df_1['spread_std'] - df_1['spread_std'].mean()) / df_1['spread_std'].std()

        # backtest_sharpe = backtest_sharpes[pair_ind]
        backtest_sharpe = backtest_results[(pair[0], pair[1])][2]
        backtest_profitable = 1 if backtest_sharpe > 0 else 0

        print(pair)
        trade_sharpe = trade_results[(pair[0], pair[1])][2]
        trade_profitable = 1 if trade_sharpe > 0 else 0

        fig, axes = plt.subplots(2, 1, squeeze=False)
        ax = axes[0, 0]
        ax.plot(df_1['spread'], c='blue')
        ax.plot(df_1['spread_mean'], c='orange')
        ax.plot(df_1['spread_mean'] + 2 * df_1['spread_std'], c='green')
        ax.plot(df_1['spread_mean'] - 2 * df_1['spread_std'], c='green')
        ax.grid()
        ax.set_title(f'Sharpe {backtest_sharpe}')
        ax = axes[1, 0]
        ax.plot(df_1['spread_norm'], c='blue')
        ax.plot(df_1['spread_norm_mean'], c='orange')
        ax.plot(df_1['spread_norm_mean'] + 2 * df_1['spread_norm_std'], c='green')
        ax.plot(df_1['spread_norm_mean'] - 2 * df_1['spread_norm_std'], c='green')
        ax.grid()
        plt.savefig(out_dir / f'backtest_spread_{ticker_1}-{ticker_2}.png')
        plt.close(fig)

        df_1 = df_1.dropna()
        results_str = f'{current_date}, {ticker_1}, {ticker_2}, '
        spread_norm_means = list(df_1['spread_norm_mean'])
        spread_norm_stds = list(df_1['spread_norm_std'])
        for spread_mean in spread_norm_means:
            results_str += f'{spread_mean}, '
        for spread_std in spread_norm_stds:
            results_str += f'{spread_std}, '
        results_str += f'{backtest_sharpe}, {backtest_profitable}, {trade_sharpe}, {trade_profitable}'
        backtest_results_file.write(f'{results_str}\n')
        backtest_results_file.flush()


def run_walk_forward(start_date, end_date,
                     n_lookback_days, n_backtest_days, n_trade_days,
                     n_pairs_vae, n_pairs_backtest,
                     returns_lookback, vae, ticker_files, fundamentals_file, out_dir,
                     r_script_exe, r_backtest_script, r_trade_script, backtest_sd,
                     datetime_format='%Y-%m-%d', cuda=False):
    """
    Runs walk forward starting at the given date, for some periods.
    """
    current_date = start_date

    # Read the fundamentals file, so we have access to sector and industry.
    fundamentals_df = pd.read_csv(fundamentals_file, index_col='ticker')

    # A file to record backtest data and outcomes, for training a predictor.
    backtest_results_file = open(Path(out_dir) / 'backtest_results.csv', 'w')

    # Loop over walk forward periods.
    while True:
        print(f'Starting walk-forward at date {current_date}')

        current_out_dir = Path(out_dir) / current_date.strftime("%Y-%m-%d")
        current_out_dir.mkdir(exist_ok=True)

        # Get the series for each ticker file.
        vae_date = current_date
        x_all, x_mean_all, x_std_all, filenames_all, lookback_start_dates = get_returns_series(
            ticker_files, vae_date, n_lookback_days, returns_lookback, datetime_format)

        # Select pairs using the VAE.
        pairs_vae, pairs_distances_vae = select_pairs_vae(x_all, vae, n_pairs_vae, fundamentals_df, cuda)
        print(f'Selected {len(pairs_vae)} pairs using VAE.')

        # Plot the pairs we selected.
        plot_vae_selections(pairs_vae, pairs_distances_vae, x_all, x_mean_all, x_std_all, fundamentals_df, current_out_dir)

        # Run backtests on the pairs selected using the VAE.
        backtest_start_date = current_date - timedelta(int(n_backtest_days * 7. / 5))  # Assume 5 trade days per week.
        backtest_results = backtest_pairs_bulk(
            pairs_vae, backtest_start_date, current_date, current_out_dir,
            r_script_exe, r_backtest_script, backtest_sd)

        # Plot the backtest results.
        fig, axes = plot_backtest_results(backtest_results, hist=False)
        plt.savefig(current_out_dir / 'backtest_results.png')
        plt.close(fig)

        # Select pairs to trade.
        pairs_backtest = select_pairs(backtest_results, pairs_vae, n_pairs_backtest)
        print(f'Selected {len(pairs_backtest)} pairs using backtest.')

        # "Trade" the selected stocks over the next period.
        trade_end_date = current_date + timedelta(int(n_trade_days * 7. / 5))  # Assume 5 trade days per week.
        print(f'Trading {len(pairs_backtest)} pairs from {current_date} to {trade_end_date}')
        trade_results = trade_pairs_bulk(pairs_backtest, current_date, trade_end_date, current_out_dir,
                                         r_script_exe, r_trade_script, backtest_sd)

        # Plot the trade results.
        fig, axes = plot_backtest_results(trade_results, hist=False)
        plt.savefig(current_out_dir / 'trade_results.png')
        plt.close(fig)

        if 0:
            # Plot the rolling mean of the spread, for each pair that we backtested.
            # This is for generating features and targets for a supervised "profitability" predictor.
            # This requires the traded pairs to be the same as the backtested pairs (in config file).
            output_spreads(pairs_vae, filenames_all, current_date, n_backtest_days, returns_lookback, backtest_results,
                           trade_results, backtest_results_file, current_out_dir, datetime_format)

        # Move forward to the next period
        current_date = trade_end_date

        if current_date > end_date:
            break

    backtest_results_file.close()

    print('Finished walk forward.')


def test():
    parser = argparse.ArgumentParser(description='Train VAE.')
    parser.add_argument('-c', '--config', help='Config file.')
    args = parser.parse_args()
    print(args)
    c = json.load(open(args.config))
    print(c)

    # clear param store
    pyro.clear_param_store()

    lookback = 50  # 160
    input_dim = 1
    returns_lookback = 20

    start_date = datetime.strptime(c['start_date'], '%Y/%m/%d') if c['start_date'] else None
    end_date = datetime.strptime(c['end_date'], '%Y/%m/%d') if c['end_date'] else None
    max_n_files = None

    out_path = Path(c['out_dir'])
    out_path.mkdir(exist_ok=True)

    # setup the VAE
    vae = VAE(c['vae_series_length'], z_dim=c['z_dim'], use_cuda=c['cuda'])

    if c['checkpoint_load']:
        checkpoint = torch.load(c['checkpoint_load'])
        vae.load_state_dict(checkpoint['model_state_dict'])

    ticker_files = glob(str(Path(c['in_dir']) / '*.csv'))
    if 1:
        ticker_files = ticker_files[:100]
    print(f'Found {len(ticker_files)} ticker files.')

    run_walk_forward(
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
        cuda=c['cuda'])


if __name__ == '__main__':
    test()
