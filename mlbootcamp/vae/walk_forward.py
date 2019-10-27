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
# import visdom
from torch.utils.data import DataLoader

import pyro
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam

from dataset import create_ticker_dataset, read_csv
from model import VAE


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


def select_next_period(z_encodings, filenames, n_pairs):
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

        selected_pairs.append(pair)

        if len(selected_pairs) == n_pairs:
            break
    selected_pairs = np.array(selected_pairs)

    return selected_pairs


def trade_next_period():
    """
    Trades the selected stocks in the next period.
    """
    pass


def get_stock_series(filename, start_date, period):
    pass


def get_ticker_from_filename(filename):
    ticker = os.path.splitext(os.path.basename(filename))[0].split('_')[0]
    return ticker


def select_pairs_vae(x_all, filenames_all, vae, n_pairs, cuda):
    z_all = []
    x_reconst_all = []
    for x in x_all:
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

        z_all.append(z_loc)
        x_reconst_all.append(x_reconst[0])

    if 0:
        # Plot the first reconstruction.
        fig, axes = plt.subplots(2, 1, squeeze=False)
        ax = axes[0, 0]
        ax.plot(x_all[0])
        ax.plot(x_reconst_all[0])
        plt.show()

    z_all = np.concatenate(z_all, axis=0)

    # Use the z-encodings to select pairs to trade.
    pairs_indices = select_next_period(z_all, filenames_all, n_pairs)

    return pairs_indices


def select_pairs_backtest(pairs_indices, tickers_all, lookback_start_dates, end_date, n_pairs,
                          r_script_exe, r_backtest_script, backtest_sd, backtest_returns_file, backtest_plot_file):

    # The backtest result for each pair.
    backtest_results = []
    for pair_ind, pair in enumerate(pairs_indices):
        # print(tickers_all[pair[0]], tickers_all[pair[1]])

        ticker_1 = tickers_all[pair[0]]
        ticker_2 = tickers_all[pair[1]]
        backtest_start_date = lookback_start_dates[pair[0]]
        backtest_end_date = end_date

        # Run a backtest.
        backtest_command = [
            r_script_exe,
            r_backtest_script,
            ticker_1,
            ticker_2,
            backtest_start_date.strftime("%Y-%m-%d"),
            backtest_end_date.strftime("%Y-%m-%d"),
            str(backtest_sd),
            backtest_returns_file,
            backtest_plot_file]
        print(f'Running selection backtest with command: {backtest_command}')
        subprocess.call(backtest_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Get the results.
        backtest_results_df = pd.read_csv(backtest_returns_file)
        backtest_return = backtest_results_df.iloc[0][1]
        backtest_stddev = backtest_results_df.iloc[1][1]
        backtest_sharpe = backtest_results_df.iloc[2][1]
        # print(backtest_results_df)

        backtest_results.append(backtest_sharpe)

    # Get the indices of the sorted backtest results.
    sorted_indices = np.flip(np.argsort(backtest_results))
    sorted_indices = sorted_indices[:n_pairs]

    # Get the best backtest pairs.
    selected_pairs_indices = []
    for i in sorted_indices:
        selected_pairs_indices.append(pairs_indices[i])
    selected_pairs_indices = np.array(selected_pairs_indices)

    return selected_pairs_indices


def select_pairs_backtest_bulk(pairs_indices, tickers_all, start_date, end_date, n_pairs, out_dir,
                          r_script_exe, r_backtest_script, backtest_sd, backtest_returns_file, backtest_plot_file):

    # Write the selected pairs out to a file.
    vae_pairs_filename = out_dir / 'vae_pairs.csv'
    vae_pairs_file = open(vae_pairs_filename, 'w')
    backtest_returns_files = []
    for pair in pairs_indices:
        ticker_1 = tickers_all[pair[0]]
        ticker_2 = tickers_all[pair[1]]
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
    backtest_results = []
    for backtest_returns_file in backtest_returns_files:
        backtest_results_df = pd.read_csv(backtest_returns_file)
        backtest_return = backtest_results_df.iloc[0][1]
        backtest_stddev = backtest_results_df.iloc[1][1]
        backtest_sharpe = backtest_results_df.iloc[2][1]
        backtest_results.append(backtest_sharpe)

    # # The backtest result for each pair.
    # backtest_results = []
    # for pair_ind, pair in enumerate(pairs_indices):
    #     # print(tickers_all[pair[0]], tickers_all[pair[1]])
    #
    #     ticker_1 = tickers_all[pair[0]]
    #     ticker_2 = tickers_all[pair[1]]
    #     backtest_start_date = lookback_start_dates[pair[0]]
    #     backtest_end_date = end_date
    #
    #     # Run a backtest.
    #     backtest_command = [
    #         r_script_exe,
    #         r_backtest_script,
    #         ticker_1,
    #         ticker_2,
    #         backtest_start_date.strftime("%Y-%m-%d"),
    #         backtest_end_date.strftime("%Y-%m-%d"),
    #         str(backtest_sd),
    #         backtest_returns_file,
    #         backtest_plot_file]
    #     print(f'Running selection backtest with command: {backtest_command}')
    #     subprocess.call(backtest_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    #
    #     # Get the results.
    #     backtest_results_df = pd.read_csv(backtest_returns_file)
    #     backtest_return = backtest_results_df.iloc[0][1]
    #     backtest_stddev = backtest_results_df.iloc[1][1]
    #     backtest_sharpe = backtest_results_df.iloc[2][1]
    #     # print(backtest_results_df)
    #
    #     backtest_results.append(backtest_sharpe)

    # Get the indices of the sorted backtest results.
    sorted_indices = np.flip(np.argsort(backtest_results))
    sorted_indices = sorted_indices[:n_pairs]

    # Get the best backtest pairs.
    selected_pairs_indices = []
    for i in sorted_indices:
        selected_pairs_indices.append(pairs_indices[i])
    selected_pairs_indices = np.array(selected_pairs_indices)

    return selected_pairs_indices


def trade_pairs(pairs_indices, tickers_all, start_date, end_date, out_dir,
                r_script_exe, r_script, backtest_sd, trade_returns_file, trade_plot_file):

    # The backtest result for each pair.
    backtest_results = {}
    for pair_ind, pair in enumerate(pairs_indices):
        # print(tickers_all[pair[0]], tickers_all[pair[1]])

        ticker_1 = tickers_all[pair[0]]
        ticker_2 = tickers_all[pair[1]]
        backtest_start_date = start_date
        backtest_end_date = end_date

        # Run a backtest.
        backtest_command = [
            r_script_exe,
            r_script,
            ticker_1,
            ticker_2,
            backtest_start_date.strftime("%Y-%m-%d"),
            backtest_end_date.strftime("%Y-%m-%d"),
            str(backtest_sd),
            trade_returns_file,
            trade_plot_file]
        print(f'Running trading backtest with command: {backtest_command}')
        subprocess.call(backtest_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Get the results.
        backtest_results_df = pd.read_csv(trade_returns_file)
        backtest_return = backtest_results_df.iloc[0][1]
        backtest_stddev = backtest_results_df.iloc[1][1]
        backtest_sharpe = backtest_results_df.iloc[2][1]
        # print(backtest_results_df)

        # Copy the backtest files so they don't get overwritten.
        new_backtest_returns_file = os.path.splitext(os.path.basename(trade_returns_file))[0]
        new_backtest_returns_file = Path(out_dir) / (str(new_backtest_returns_file) + f'_{ticker_1}-{ticker_2}.csv')
        shutil.copy(trade_returns_file, new_backtest_returns_file)

        new_backtest_plot_file = os.path.splitext(os.path.basename(trade_plot_file))[0]
        new_backtest_plot_file = Path(out_dir) / (str(new_backtest_plot_file) + f'_{ticker_1}-{ticker_2}.jpeg')
        shutil.copy(trade_plot_file, new_backtest_plot_file)

        backtest_results[tuple(pair)] = backtest_sharpe

    return backtest_results


def trade_pairs_bulk(pairs_indices, tickers_all, start_date, end_date, out_dir,
                r_script_exe, r_script, backtest_sd, trade_returns_file, trade_plot_file):

    # Write the selected pairs out to a file.
    pairs_filename = out_dir / 'vae_pairs_trade.csv'
    pairs_file = open(pairs_filename, 'w')
    trade_returns_files = []
    for pair in pairs_indices:
        ticker_1 = tickers_all[pair[0]]
        ticker_2 = tickers_all[pair[1]]
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
    results = []
    backtest_results = {}
    for i, backtest_returns_file in enumerate(trade_returns_files):
        backtest_results_df = pd.read_csv(backtest_returns_file)
        backtest_return = backtest_results_df.iloc[0][1]
        backtest_stddev = backtest_results_df.iloc[1][1]
        backtest_sharpe = backtest_results_df.iloc[2][1]
        results.append(backtest_sharpe)

        pair = pairs_indices[i]
        backtest_results[tuple(pair)] = [backtest_return, backtest_stddev, backtest_sharpe]


    # # The backtest result for each pair.
    # backtest_results = {}
    # for pair_ind, pair in enumerate(pairs_indices):
    #     # print(tickers_all[pair[0]], tickers_all[pair[1]])
    #
    #     ticker_1 = tickers_all[pair[0]]
    #     ticker_2 = tickers_all[pair[1]]
    #     backtest_start_date = start_date
    #     backtest_end_date = end_date
    #
    #     # Run a backtest.
    #     backtest_command = [
    #         r_script_exe,
    #         r_script,
    #         ticker_1,
    #         ticker_2,
    #         backtest_start_date.strftime("%Y-%m-%d"),
    #         backtest_end_date.strftime("%Y-%m-%d"),
    #         str(backtest_sd),
    #         trade_returns_file,
    #         trade_plot_file]
    #     print(f'Running trading backtest with command: {backtest_command}')
    #     subprocess.call(backtest_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    #
    #     # Get the results.
    #     backtest_results_df = pd.read_csv(trade_returns_file)
    #     backtest_return = backtest_results_df.iloc[0][1]
    #     backtest_stddev = backtest_results_df.iloc[1][1]
    #     backtest_sharpe = backtest_results_df.iloc[2][1]
    #     # print(backtest_results_df)
    #
    #     # Copy the backtest files so they don't get overwritten.
    #     new_backtest_returns_file = os.path.splitext(os.path.basename(trade_returns_file))[0]
    #     new_backtest_returns_file = Path(out_dir) / (str(new_backtest_returns_file) + f'_{ticker_1}-{ticker_2}.csv')
    #     shutil.copy(trade_returns_file, new_backtest_returns_file)
    #
    #     new_backtest_plot_file = os.path.splitext(os.path.basename(trade_plot_file))[0]
    #     new_backtest_plot_file = Path(out_dir) / (str(new_backtest_plot_file) + f'_{ticker_1}-{ticker_2}.jpeg')
    #     shutil.copy(trade_plot_file, new_backtest_plot_file)
    #
    #     backtest_results[tuple(pair)] = backtest_sharpe

    return backtest_results


def run_walk_forward(start_date, end_date, n_lookback_days, n_trade_days, n_pairs_vae, n_pairs_backtest,
                     returns_lookback, vae, ticker_files, out_dir,
                     r_script_exe, r_backtest_script, r_trade_script, backtest_sd,
                     backtest_returns_file, backtest_plot_file, trade_returns_file, trade_plot_file,
                     datetime_format='%Y-%m-%d', cuda=False):
    """
    Runs walk forward starting at the given date, for some periods.
    """
    current_date = start_date

    # Loop over walk forward periods.
    trade_sharpes = []
    while True:
        print(f'Starting walk-forward at date {current_date}')

        current_out_dir = Path(out_dir) / current_date.strftime("%Y-%m-%d")
        current_out_dir.mkdir(exist_ok=True)

        # Get the series for each ticker file.
        x_all = []
        filenames_all = []
        tickers_all = []
        lookback_start_dates = []
        for file in ticker_files:
            df = read_csv(file, datetime_format)

            # Get the lookback period ending at the current date.
            df = df.loc[:current_date]
            df = df.iloc[-(returns_lookback + n_lookback_days):]

            # Calculate normalised returns.
            df['returns'] = df['close'].pct_change()
            df['returns'] = \
                (df['returns'] - df['returns'].rolling(returns_lookback).mean()) / df['returns'].rolling(returns_lookback).std()
            df = df.dropna()

            # We should be left with the number of lookback days.
            if df.shape[0] != n_lookback_days:
                continue

            x = np.array(df['returns'])

            x_all.append(x)
            filenames_all.extend(file)
            tickers_all.append(get_ticker_from_filename(file))
            lookback_start_dates.append(df.index[0])

        # Filter using the VAE.
        pairs_indices_vae = select_pairs_vae(x_all, filenames_all, vae, n_pairs_vae, cuda)
        print(f'Selected {len(pairs_indices_vae)} pairs using VAE.')
        if 0:
            # Plot the pairs we selected.
            n = min(10, n_pairs_vae)
            fig, axes = plt.subplots(n, 1, squeeze=False)
            for pair_ind, pair in enumerate(pairs_indices_vae[:n]):
                ax = axes[pair_ind, 0]
                ax.plot(x_all[pair[0]], label=tickers_all[pair[0]])
                ax.plot(x_all[pair[1]], label=tickers_all[pair[1]])
                ax.legend(loc='upper left')
            plt.show()

        # # Write the selected pairs out to a file.
        # vae_pairs_filename = current_out_dir / 'vae_pairs.csv'
        # vae_pairs_file = open(vae_pairs_filename, 'wb')
        # for pair in pairs_indices_vae:
        #     vae_pairs_file.write(f'{tickers_all[pair[0]]},{tickers_all[pair[1]]}\n')
        # vae_pairs_file.close()

        # Filter using a backtest.
        if 0:
            pairs_indices_backtest = select_pairs_backtest(
                pairs_indices_vae, tickers_all, lookback_start_dates, current_date, n_pairs_backtest,
                r_script_exe, r_backtest_script, backtest_sd, backtest_returns_file, backtest_plot_file)
            print(f'Selected {len(pairs_indices_backtest)} pairs using backtest.')
        else:
            backtest_start_date = current_date - timedelta(int(n_trade_days * 7 / 5))  # Assume 5 trade days per week.
            pairs_indices_backtest = select_pairs_backtest_bulk(
                pairs_indices_vae, tickers_all, backtest_start_date, current_date, n_pairs_backtest, current_out_dir,
                r_script_exe, r_backtest_script, backtest_sd, backtest_returns_file, backtest_plot_file)
            print(f'Selected {len(pairs_indices_backtest)} pairs using backtest.')

        if 0:
            # Plot the pairs we selected.
            n = min(10, n_pairs_backtest)
            fig, axes = plt.subplots(n, 1, squeeze=False)
            for pair_ind, pair in enumerate(pairs_indices_backtest[:n]):
                ax = axes[pair_ind, 0]
                ax.plot(x_all[pair[0]], label=tickers_all[pair[0]])
                ax.plot(x_all[pair[1]], label=tickers_all[pair[1]])
                ax.legend(loc='upper left')
            plt.show()

        # "Trade" the selected stocks over the next period.
        trade_end_date = current_date + timedelta(int(n_trade_days * 7 / 5))  # Assume 5 trade days per week.
        print(f'Trading {len(pairs_indices_backtest)} pairs from {current_date} to {trade_end_date}')
        trade_results = trade_pairs_bulk(
            pairs_indices_backtest, tickers_all, current_date, current_date + timedelta(3*365/12), current_out_dir,
            r_script_exe, r_trade_script, backtest_sd, trade_returns_file, trade_plot_file)
        print(f'Results of trade at date {current_date}: {trade_results}')

        sharpes = [trade_results[pair][2] for pair in trade_results]
        trade_sharpes.extend(sharpes)
        fig, axes = plt.subplots(1, 1, squeeze=False)
        ax = axes[0, 0]
        ax.hist(trade_sharpes)
        mu = np.mean(trade_sharpes)
        sd = np.std(trade_sharpes)
        x = np.linspace(min(trade_sharpes), max(trade_sharpes), 100)
        ax2 = ax.twinx()
        ax2.plot(x, stats.norm.pdf(x, mu, sd), c='r')
        plt.savefig(current_out_dir / 'sharpes_current.png')
        plt.close(fig)

        # Move forward to the next period
        current_date = trade_end_date

        if current_date > end_date:
            break

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
    returns_lookback = 50

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
    if 0:
        ticker_files = ticker_files[:10]
    print(f'Found {len(ticker_files)} ticker files.')

    run_walk_forward(
        start_date=start_date,
        end_date=end_date,
        n_lookback_days=c['n_lookback_days'],
        n_trade_days=c['n_trade_days'],
        n_pairs_vae=c['n_pairs_vae'],
        n_pairs_backtest=c['n_pairs_backtest'],
        vae=vae,
        returns_lookback=returns_lookback,
        ticker_files=ticker_files,
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
    test()
