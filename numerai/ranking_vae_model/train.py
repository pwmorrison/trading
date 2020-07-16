import argparse
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam

from dataset import create_ranking_dataset
from model import VAE


def train():
    parser = argparse.ArgumentParser(description='Train VAE.')
    parser.add_argument('-c', '--config', default='train_config.json', help='Config file.')
    args = parser.parse_args()
    print(args)
    c = json.load(open(args.config))
    print(c)

    # clear param store
    pyro.clear_param_store()

    input_dim = 1
    max_n_examples = None

    out_path = Path(c['out_dir'])
    out_path.mkdir(exist_ok=True)

    dataset_train = create_ranking_dataset(c['training_filename'], 0.0, 0.7, max_n_examples=max_n_examples)
    dataset_val = create_ranking_dataset(c['training_filename'], 0.7, 1.0, max_n_examples=max_n_examples)
    train_loader = DataLoader(dataset_train, batch_size=c['batch_size'], shuffle=True, num_workers=3, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=c['batch_size'], shuffle=False, num_workers=3, drop_last=True)

    N_train_data = len(dataset_train)
    N_val_data = len(dataset_val)
    N_mini_batches = N_train_data // c['batch_size']
    N_train_time_slices = c['batch_size'] * N_mini_batches

    print(f'N_train_data: {N_train_data}, N_val_data: {N_val_data}')

    # setup the VAE
    vae = VAE(c['num_features'], z_dim=c['z_dim'], use_cuda=c['cuda'])

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
    num_epochs = c['n_epochs']
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch} of {num_epochs}.')
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for batch_num, batch in enumerate(train_loader):
            print(f'Batch {batch_num} of {N_mini_batches}.')
            features_1 = batch['features_1']
            features_2 = batch['features_2']
            target_class = batch['target_class']
            if c['cuda']:
                features_1 = features_1.cuda()
                features_2 = features_2.cuda()
                target_class = target_class.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(features_1.float(), features_2.float(), target_class.float())

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
            print('Evaluation validation data.')
            val_loss = 0.
            true_positives = 0
            num_val_examples = 0

            # Compute the loss over the validation set.
            for i, batch in enumerate(val_loader):
                features_1 = batch['features_1']
                features_2 = batch['features_2']
                target_class = batch['target_class']
                # if on GPU put mini-batch into CUDA memory
                if c['cuda']:
                    features_1 = features_1.cuda()
                    features_2 = features_2.cuda()
                    target_class = target_class.cuda()
                features_1 = features_1.float()
                features_2 = features_2.float()
                target_class = target_class.float()
                # compute ELBO estimate and accumulate loss
                val_loss += svi.evaluate_loss(features_1, features_2, target_class)

                if 1:
                    z_1_loc, z_1_scale, z_1_sample = vae.encode_x(features_1)
                    z_2_loc, z_2_scale, z_2_sample = vae.encode_x(features_2)
                    pred = torch.sigmoid(z_1_loc - z_2_loc)
                    pred[pred >= 0.5] = 1
                    pred[pred < 0.5] = 0
                    target_class = torch.unsqueeze(target_class, dim=1)
                    pred_target = torch.cat([pred, target_class], dim=1)
                    # print(f'pred vs target: {pred_target}')

                    pred = pred.cpu().detach().numpy()
                    target_class = target_class.cpu().detach().numpy()
                    pred_correct = (pred == target_class).astype(int)
                    # print(pred_correct)
                    # print(target_class.shape)
                    pred_correct.flatten()
                    true_positives += np.sum(pred_correct)
                    num_val_examples += pred_correct.shape[0]
                    # accuracy = np.sum(pred_correct) / pred_correct.shape[0]
                    # print('accuracy:', accuracy)
                    # return

                if 0:
                    # Make some rank predictions.
                    z_1_loc, z_1_scale, z_1_sample = vae.encode_x(x_1)
                    z_2_loc, z_2_scale, z_2_sample = vae.encode_x(x_2)
                    z_loc = torch.cat([z_1_loc, z_2_loc], dim=1)
                    print(f'y: {y}, z_loc: {z_loc}')
                    return

                if 0:
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
                        plt.close(fig)

            # report test diagnostics
            normalizer_val = len(val_loader.dataset)
            total_epoch_loss_val = val_loss / normalizer_val
            val_elbo.append(total_epoch_loss_val)
            print("[epoch %03d]  average val loss: %.4f" % (epoch, total_epoch_loss_val))

            accuracy = true_positives / num_val_examples
            print(f'Accuracy: {accuracy}')


    print('Finished training.')


if __name__ == '__main__':
    train()
