import argparse

import numpy as np
import datetime
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


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        # setup the three linear transformations used
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, self.input_dim)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, output_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        # loc_img = torch.sigmoid(self.fc21(hidden))
        loc_img = self.fc21(hidden)
        return loc_img


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, x_dim, z_dim=50, hidden_dim=400, use_cuda=False):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(x_dim, z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim, x_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.x_dim = x_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images
            # pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, self.x_dim))
            pyro.sample("obs", dist.Normal(loc_img, 0.1).to_event(1), obs=x.reshape(-1, self.x_dim))
            # return the loc so we can visualize it later
            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img

    def encode_x(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()

        return z_loc, z_scale, z


def find_similar(vae, dataloader):
    # Find the latent space vector for every example in the test set.
    x_all = []
    z_all = []
    x_reconst_all = []
    for x in dataloader:
        # x, z, x_reconst = test_minibatch(dmm, test_batch, args, sample_z=True)
        if args.cuda:
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
    x_all = np.concatenate(x_all, axis=0)
    z_all = np.concatenate(z_all, axis=0)
    x_reconst_all = np.concatenate(x_reconst_all, axis=0)

    # Get the closest latent to the query.
    n_neighbours = 5
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1 + n_neighbours, algorithm='ball_tree').fit(z_all)
    distances, indices = nbrs.kneighbors(z_all)

    # Select a random latent.
    query_index = 0
    # Skip the first closest index, since it is just the query index.
    closest_indices = indices[query_index][1:]

    # Plot the query and closest series.
    fig, axes = plt.subplots(1 + n_neighbours, 1, squeeze=False)
    ax = axes[0, 0]
    x_series = x_all[query_index, ...]
    ax.plot(range(x_series.shape[0]), x_series, c='r')
    ax.grid()
    for i in range(n_neighbours):
        ax = axes[i + 1, 0]
        x_series = x_all[closest_indices[i], ...]
        ax.plot(range(x_series.shape[0]), x_series, c='b')
        ax.grid()
    plt.show()

    return


def main(args):
    # clear param store
    pyro.clear_param_store()

    batch_size = 64
    root_dir = r'D:\projects\trading\mlbootcamp\tickers'
    series_length = 60
    lookback = 50  # 160
    input_dim = 1
    train_start_date = datetime.date(2010, 1, 1)
    train_end_date = datetime.date(2016, 1, 1)
    test_start_date = train_end_date
    test_end_date = datetime.date(2019, 1, 1)
    min_sequence_length_train = 2 * (series_length + lookback)
    min_sequence_length_test = 2 * (series_length + lookback)

    load_path = 'out/checkpoint_0035.pt'

    # setup MNIST data loaders
    # train_loader, test_loader
    # train_loader, test_loader = setup_data_loaders(MNIST, use_cuda=args.cuda, batch_size=256)
    dataset_train = create_ticker_dataset(root_dir, series_length, lookback, min_sequence_length_train,
                                          start_date=train_start_date, end_date=train_end_date)
    dataset_test = create_ticker_dataset(root_dir, series_length, lookback, min_sequence_length_test,
                                         start_date=test_start_date, end_date=test_end_date, fixed_start_date=True)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    N_train_data = len(dataset_train)
    N_test_data = len(dataset_test)
    N_mini_batches = N_train_data // batch_size
    N_train_time_slices = batch_size * N_mini_batches

    print(f'N_train_data: {N_train_data}, N_test_data: {N_test_data}')

    # setup the VAE
    vae = VAE(series_length, use_cuda=args.cuda)

    # setup the optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    if load_path:
        checkpoint = torch.load(load_path)
        vae.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if args.cuda:
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
        }, f'out/checkpoint_{epoch:04d}.pt')

        if epoch % args.test_frequency == 0:
            # initialize loss accumulator
            test_loss = 0.
            # compute the loss over the entire test set
            for i, x in enumerate(test_loader):
                # if on GPU put mini-batch into CUDA memory
                if args.cuda:
                    x = x.cuda()
                x = x.float()
                # compute ELBO estimate and accumulate loss
                test_loss += svi.evaluate_loss(x)

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
                    fig.savefig(f'out/val_{epoch:03d}.png')

            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo.append(total_epoch_loss_test)
            print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))

            # t-SNE.
            all_z_latents = []
            for x in test_loader:
                # z_latents = minibatch_inference(dmm, test_batch)
                # z_latents = encode_x_to_z(dmm, test_batch, sample_z_t=False)
                # x, z, x_reconst = test_minibatch(dmm, test_batch, args, sample_z=True)

                if args.cuda:
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

            fig.savefig(f'out/tsne_{epoch:03d}.png')
            plt.close(fig)

    if 1:
        find_similar(vae, test_loader)

    # Visualise first batch.
    x = next(iter(test_loader))
    if args.cuda:
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
    fig.savefig(f'val_test.png')

    return vae


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.4.1')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=0, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=5, type=int, help='how often we evaluate the test set')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=True, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    parser.add_argument('-visdom', '--visdom_flag', action="store_true", help='Whether plotting in visdom is desired')
    parser.add_argument('-i-tsne', '--tsne_iter', default=0, type=int, help='epoch when tsne visualization runs')
    args = parser.parse_args()

    model = main(args)
