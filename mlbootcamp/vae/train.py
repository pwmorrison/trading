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
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam
from utils.mnist_cached import MNISTCached as MNIST
from utils.mnist_cached import setup_data_loaders
from utils.vae_plots import mnist_test_tsne, plot_llk, plot_vae_samples

from dataset import create_ticker_dataset
from model import VAE


def train():
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

    train_start_date = datetime.strptime(c['train_start_date'], '%Y/%m/%d')
    train_end_date = datetime.strptime(c['train_end_date'], '%Y/%m/%d')
    val_start_date = datetime.strptime(c['val_start_date'], '%Y/%m/%d')
    val_end_date = datetime.strptime(c['val_end_date'], '%Y/%m/%d')
    min_sequence_length_train = 2 * (c['series_length'] + lookback)
    min_sequence_length_test = 2 * (c['series_length'] + lookback)
    max_n_files = None

    out_path = Path(c['out_dir'])
    out_path.mkdir(exist_ok=True)

    dataset_train = create_ticker_dataset(c['in_dir'], c['series_length'], lookback, min_sequence_length_train,
                                          start_date=train_start_date, end_date=train_end_date,
                                          normalised_returns=c['normalised_returns'], max_n_files=max_n_files)
    dataset_val = create_ticker_dataset(c['in_dir'], c['series_length'], lookback, min_sequence_length_test,
                                        start_date=val_start_date, end_date=val_end_date, fixed_start_date=True,
                                        normalised_returns=c['normalised_returns'], max_n_files=max_n_files)
    train_loader = DataLoader(dataset_train, batch_size=c['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=c['batch_size'], shuffle=False, num_workers=0, drop_last=True)

    N_train_data = len(dataset_train)
    N_val_data = len(dataset_val)
    N_mini_batches = N_train_data // c['batch_size']
    N_train_time_slices = c['batch_size'] * N_mini_batches

    print(f'N_train_data: {N_train_data}, N_val_data: {N_val_data}')

    # setup the VAE
    vae = VAE(c['series_length'], z_dim=c['z_dim'], use_cuda=c['cuda'])

    # setup the optimizer
    adam_args = {"lr": c['learning_rate']}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    elbo = JitTrace_ELBO() if c['jit'] else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    if c['checkpoint_load']:
        checkpoint = torch.load(c['checkpoint_load'])
        vae.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    train_elbo = []
    val_elbo = []
    # training loop
    for epoch in range(c['n_epochs']):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for batch in train_loader:
            x = batch['series']
            # if on GPU put mini-batch into CUDA memory
            if c['cuda']:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x.float())

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        torch.save({
            'epoch': epoch,
            'model_state_dict': vae.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss
        }, out_path / c['checkpoint_save'].format(epoch))

        if epoch % c['val_frequency'] == 0:
            # initialize loss accumulator
            val_loss = 0.
            # compute the loss over the entire test set
            for i, batch in enumerate(val_loader):
                x = batch['series']
                # if on GPU put mini-batch into CUDA memory
                if c['cuda']:
                    x = x.cuda()
                x = x.float()
                # compute ELBO estimate and accumulate loss
                val_loss += svi.evaluate_loss(x)

                if i == 0:
                    # Visualise first batch.
                    x_reconst = vae.reconstruct_img(x)
                    x = x.cpu().numpy()
                    x_reconst = x_reconst.cpu().detach().numpy()

                    n = min(5, x.shape[0])
                    fig, axes = plt.subplots(n, 1, squeeze=False)
                    for s in range(n):
                        ax = axes[s, 0]
                        ax.plot(x[s])
                        ax.plot(x_reconst[s])
                    fig.savefig(out_path / f'val_{epoch:03d}.png')

            # report test diagnostics
            normalizer_val = len(val_loader.dataset)
            total_epoch_loss_val = val_loss / normalizer_val
            val_elbo.append(total_epoch_loss_val)
            print("[epoch %03d]  average val loss: %.4f" % (epoch, total_epoch_loss_val))

            # t-SNE.
            all_z_latents = []
            for batch in val_loader:
                x = batch['series']
                # z_latents = minibatch_inference(dmm, test_batch)
                # z_latents = encode_x_to_z(dmm, test_batch, sample_z_t=False)
                # x, z, x_reconst = test_minibatch(dmm, test_batch, args, sample_z=True)

                if c['cuda']:
                    x = x.cuda()

                z_loc, z_scale, z = vae.encode_x(x.float())
                all_z_latents.append(z.cpu().numpy())

            # all_latents = torch.cat(all_z_latents, dim=0)
            all_latents = np.concatenate(all_z_latents, axis=0)

            # Run t-SNE with 2 output dimensions.
            from sklearn.manifold import TSNE
            model_tsne = TSNE(n_components=2, random_state=0)
            # z_states = all_latents.detach().cpu().numpy()
            z_states = all_latents
            z_embed = model_tsne.fit_transform(z_states)
            # Plot t-SNE embedding.
            fig = plt.figure()
            plt.scatter(z_embed[:, 0], z_embed[:, 1], s=10)

            fig.savefig(out_path / f'tsne_{epoch:03d}.png')
            plt.close(fig)

    print('Finished training.')


if __name__ == '__main__':
    train()
