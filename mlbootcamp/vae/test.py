import argparse
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# import visdom
from torch.utils.data import DataLoader

import pyro
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam

from dataset import create_ticker_dataset
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


def test():
    parser = argparse.ArgumentParser(description='Train VAE.')
    parser.add_argument('-c', '--config', help='Config file.')
    args = parser.parse_args()
    print(args)
    c = json.load(open(args.config))
    print(c)

    # clear param store
    pyro.clear_param_store()

    # batch_size = 64
    # root_dir = r'D:\projects\trading\mlbootcamp\tickers'
    # series_length = 60
    lookback = 50  # 160
    input_dim = 1

    test_start_date = datetime.strptime(c['test_start_date'], '%Y/%m/%d')
    test_end_date = datetime.strptime(c['test_end_date'], '%Y/%m/%d')
    min_sequence_length_test = 2 * (c['series_length'] + lookback)
    max_n_files = None

    out_path = Path(c['out_dir'])
    out_path.mkdir(exist_ok=True)

    load_path = 'out_saved/checkpoint_0035.pt'

    dataset_test = create_ticker_dataset(c['in_dir'], c['series_length'], lookback, min_sequence_length_test,
                                         start_date=test_start_date, end_date=test_end_date, fixed_start_date=True,
                                         max_n_files=max_n_files)
    test_loader = DataLoader(dataset_test, batch_size=c['batch_size'], shuffle=False, num_workers=0, drop_last=True)

    # N_train_data = len(dataset_train)
    N_test_data = len(dataset_test)
    # N_mini_batches = N_train_data // c['batch_size']
    # N_train_time_slices = c['batch_size'] * N_mini_batches

    print(f'N_test_data: {N_test_data}')

    # setup the VAE
    vae = VAE(c['series_length'], use_cuda=c['cuda'])

    # setup the optimizer
    # adam_args = {"lr": args.learning_rate}
    # optimizer = Adam(adam_args)

    # setup the inference algorithm
    # elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    # svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    if load_path:
        checkpoint = torch.load(load_path)
        vae.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 1:
        find_similar(vae, test_loader, c['cuda'])

    # Visualise first batch.
    batch = next(iter(test_loader))
    x = batch['series']
    if c['cuda']:
        x = x.cuda()
    x = x.float()
    x_reconst = vae.reconstruct_img(x)
    x = x.cpu().numpy()
    x_reconst = x_reconst.cpu().detach().numpy()

    n = min(5, x.shape[0])
    fig, axes = plt.subplots(n, 1, squeeze=False)
    for s in range(n):
        ax = axes[s, 0]
        ax.plot(x[s])
        ax.plot(x_reconst[s])
    fig.savefig(out_path / f'test_batch.png')


if __name__ == '__main__':
    test()
