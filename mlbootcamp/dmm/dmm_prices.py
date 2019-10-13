"""
An implementation of a Deep Markov Model in Pyro based on reference [1].
This is essentially the DKS variant outlined in the paper. The primary difference
between this implementation and theirs is that in our version any KL divergence terms
in the ELBO are estimated via sampling, while they make use of the analytic formulae.
We also illustrate the use of normalizing flows in the variational distribution (in which
case analytic formulae for the KL divergences are in any case unavailable).

Reference:

[1] Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
    Rahul G. Krishnan, Uri Shalit, David Sontag
"""

import argparse
import time
from os.path import exists

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import polyphonic_data_loader as poly
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import InverseAutoregressiveFlow, TransformedDistribution
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.nn import AutoRegressiveNN
from pyro.optim import ClippedAdam
from util import get_logger

from dataset import create_ticker_dataset
from tsne import plot_tsne


class Emitter(nn.Module):
    """
    Parameterizes the observation likelihood `p(x_t | z_t)`
    """

    def __init__(self, input_dim, z_dim, emission_dim):
        super(Emitter, self).__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        ps = torch.tanh(self.lin_hidden_to_input(h2))
        return ps


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """

    def __init__(self, z_dim, transition_dim):
        super(GatedTransition, self).__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        return loc, scale


class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """

    def __init__(self, z_dim, rnn_dim):
        super(Combiner, self).__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        # return loc, scale which can be fed into Normal
        return loc, scale


class DMM(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """
    def __init__(self, input_dim=88, z_dim=50, emission_dim=50,
                 transition_dim=100, rnn_dim=300, num_layers=1, rnn_dropout_rate=0.0,
                 num_iafs=0, iaf_dim=50, use_cuda=False):
        super(DMM, self).__init__()
        # instantiate PyTorch modules used in the model and guide below
        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)
        # dropout just takes effect on inner layers of rnn
        rnn_dropout_rate = 0. if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                          batch_first=True, bidirectional=False, num_layers=num_layers,
                          dropout=rnn_dropout_rate)

        # if we're using normalizing flows, instantiate those too
        self.iafs = [InverseAutoregressiveFlow(AutoRegressiveNN(z_dim, [iaf_dim])) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()

    # the model p(x_{1:T} | z_{1:T}) p(z_{1:T})
    def model(self, mini_batch, mini_batch_reversed, mini_batch_mask,
              mini_batch_seq_lengths, annealing_factor=1.0):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))

        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("z_minibatch", len(mini_batch)):
            # sample the latents z and observed x's one time step at a time
            for t in range(1, T_max + 1):
                # the next chunk of code samples z_t ~ p(z_t | z_{t-1})
                # note that (both here and elsewhere) we use poutine.scale to take care
                # of KL annealing. we use the mask() method to deal with raggedness
                # in the observed data (i.e. different sequences in the mini-batch
                # have different lengths)

                # first compute the parameters of the diagonal gaussian distribution p(z_t | z_{t-1})
                z_loc, z_scale = self.trans(z_prev)

                # then sample z_t according to dist.Normal(z_loc, z_scale)
                # note that we use the reshape method so that the univariate Normal distribution
                # is treated as a multivariate Normal distribution with a diagonal covariance.
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t,
                                      dist.Normal(z_loc, z_scale)
                                          .mask(mini_batch_mask[:, t - 1:t])
                                          .to_event(1))

                # compute the probabilities that parameterize the bernoulli likelihood
                emission_probs_t = self.emitter(z_t)
                # the next statement instructs pyro to observe x_t according to the
                # bernoulli distribution p(x_t|z_t)
                pyro.sample("obs_x_%d" % t,
                            # dist.Bernoulli(emission_probs_t)
                            dist.Normal(emission_probs_t, 0.5)
                                .mask(mini_batch_mask[:, t - 1:t])
                                .to_event(1),
                            obs=mini_batch[:, t - 1, :])
                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev = z_t

    # the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
    def guide(self, mini_batch, mini_batch_reversed, mini_batch_mask,
              mini_batch_seq_lengths, annealing_factor=1.0):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)

        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))

        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(mini_batch)):
            # sample the latents z one time step at a time
            for t in range(1, T_max + 1):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

                # if we are using normalizing flows, we apply the sequence of transformations
                # parameterized by self.iafs to the base distribution defined in the previous line
                # to yield a transformed distribution that we use for q(z_t|...)
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                    assert z_dist.event_shape == (self.z_q_0.size(0),)
                    assert z_dist.batch_shape == (len(mini_batch),)
                else:
                    z_dist = dist.Normal(z_loc, z_scale)
                    assert z_dist.event_shape == ()
                    assert z_dist.batch_shape == (len(mini_batch), self.z_q_0.size(0))

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=annealing_factor):
                    if len(self.iafs) > 0:
                        # in output of normalizing flow, all dimensions are correlated (event shape is not empty)
                        z_t = pyro.sample("z_%d" % t,
                                          z_dist.mask(mini_batch_mask[:, t - 1]))
                    else:
                        # when no normalizing flow used, ".to_event(1)" indicates latent dimensions are independent
                        z_t = pyro.sample("z_%d" % t,
                                          z_dist.mask(mini_batch_mask[:, t - 1:t])
                                          .to_event(1))
                # the latent sampled at this time step will be conditioned upon in the next time step
                # so keep track of it
                z_prev = z_t


# saves the model and optimizer states to disk
def save_checkpoint(dmm, adam, log):
    log("saving model to %s..." % args.save_model)
    torch.save(dmm.state_dict(), args.save_model)
    log("saving optimizer states to %s..." % args.save_opt)
    adam.save(args.save_opt)
    log("done saving model and optimizer checkpoints to disk.")


# loads the model and optimizer states from disk
def load_checkpoint(dmm, adam, log):
    assert exists(args.load_opt) and exists(args.load_model), \
        "--load-model and/or --load-opt misspecified"
    log("loading model from %s..." % args.load_model)
    dmm.load_state_dict(torch.load(args.load_model))
    log("loading optimizer states from %s..." % args.load_opt)
    adam.load(args.load_opt)
    log("done loading model and optimizer states.")


# prepare a mini-batch and take a gradient step to minimize -elbo
def process_minibatch(svi, epoch, mini_batch, n_mini_batches, which_mini_batch, shuffled_indices):
    if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
        # compute the KL annealing factor approriate for the current mini-batch in the current epoch
        min_af = args.minimum_annealing_factor
        annealing_factor = min_af + (1.0 - min_af) * \
            (float(which_mini_batch + epoch * n_mini_batches + 1) /
             float(args.annealing_epochs * n_mini_batches))
    else:
        # by default the KL annealing factor is unity
        annealing_factor = 1.0

    # Generate dummy data that we can feed into the below fn.
    training_data_sequences = mini_batch.type(torch.FloatTensor)
    mini_batch_indices = torch.arange(0, training_data_sequences.size(0))
    training_seq_lengths = torch.full((training_data_sequences.size(0),), training_data_sequences.size(1)).type(torch.IntTensor)

    # grab a fully prepped mini-batch using the helper function in the data loader
    mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
        = poly.get_mini_batch(mini_batch_indices, training_data_sequences,
                              training_seq_lengths, cuda=args.cuda)

    # do an actual gradient step
    loss = svi.step(mini_batch, mini_batch_reversed, mini_batch_mask,
                    mini_batch_seq_lengths, annealing_factor)

    # keep track of the training loss
    return loss


def test_minibatch(dmm, mini_batch, args, sample_z=True):

    # Generate data that we can feed into the below fn.
    test_data_sequences = mini_batch.type(torch.FloatTensor)
    mini_batch_indices = torch.arange(0, test_data_sequences.size(0))
    test_seq_lengths = torch.full((test_data_sequences.size(0),), test_data_sequences.size(1)).type(
        torch.IntTensor)

    # grab a fully prepped mini-batch using the helper function in the data loader
    mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
        = poly.get_mini_batch(mini_batch_indices, test_data_sequences,
                              test_seq_lengths, cuda=args.cuda)

    # Get the initial RNN state.
    h_0 = dmm.h_0
    h_0_contig = h_0.expand(1, mini_batch.size(0), dmm.rnn.hidden_size).contiguous()

    # Feed the test sequence into the RNN.
    rnn_output, rnn_hidden_state = dmm.rnn(mini_batch_reversed, h_0_contig)

    # Reverse the time ordering of the hidden state and unpack it.
    rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
    # print(rnn_output)
    # print(rnn_output.shape)

    # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
    z_prev = dmm.z_q_0.expand(mini_batch.size(0), dmm.z_q_0.size(0))

    # sample the latents z one time step at a time
    T_max = mini_batch.size(1)
    sequence_z = []
    sequence_output = []
    for t in range(1, T_max + 1):
        # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
        z_loc, z_scale = dmm.combiner(z_prev, rnn_output[:, t - 1, :])

        if sample_z:
            # if we are using normalizing flows, we apply the sequence of transformations
            # parameterized by self.iafs to the base distribution defined in the previous line
            # to yield a transformed distribution that we use for q(z_t|...)
            if len(dmm.iafs) > 0:
                z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), dmm.iafs)
            else:
                z_dist = dist.Normal(z_loc, z_scale)
            assert z_dist.event_shape == ()
            assert z_dist.batch_shape == (len(mini_batch), dmm.z_q_0.size(0))

            # sample z_t from the distribution z_dist
            annealing_factor = 1.0
            with pyro.poutine.scale(scale=annealing_factor):
                z_t = pyro.sample("z_%d" % t,
                                  z_dist.mask(mini_batch_mask[:, t - 1:t])
                                  .to_event(1))
        else:
            z_t = z_loc

        z_t_np = z_t.detach().numpy()
        z_t_np = z_t_np[:, np.newaxis, :]
        sequence_z.append(z_t_np)

        # print("z_{}:".format(t), z_t)
        # print(z_t.shape)

        # compute the probabilities that parameterize the bernoulli likelihood
        emission_probs_t = dmm.emitter(z_t)

        emission_probs_t_np = emission_probs_t.detach().numpy()
        sequence_output.append(emission_probs_t_np)

        # print("x_{}:".format(t), emission_probs_t)
        # print(emission_probs_t.shape)

        # the latent sampled at this time step will be conditioned upon in the next time step
        # so keep track of it
        z_prev = z_t

    # Run the model another few steps.
    n_extra_steps = 100
    for t in range(1, n_extra_steps + 1):
        # first compute the parameters of the diagonal gaussian distribution p(z_t | z_{t-1})
        z_loc, z_scale = dmm.trans(z_prev)

        # then sample z_t according to dist.Normal(z_loc, z_scale)
        # note that we use the reshape method so that the univariate Normal distribution
        # is treated as a multivariate Normal distribution with a diagonal covariance.
        annealing_factor = 1.0
        with poutine.scale(scale=annealing_factor):
            z_t = pyro.sample("z_%d" % t,
                              dist.Normal(z_loc, z_scale)
                              # .mask(mini_batch_mask[:, t - 1:t])
                              .to_event(1))

        z_t_np = z_t.detach().numpy()
        z_t_np = z_t_np[:, np.newaxis, :]
        sequence_z.append(z_t_np)

        # compute the probabilities that parameterize the bernoulli likelihood
        emission_probs_t = dmm.emitter(z_t)

        emission_probs_t_np = emission_probs_t.detach().numpy()
        sequence_output.append(emission_probs_t_np)

        # the latent sampled at this time step will be conditioned upon
        # in the next time step so keep track of it
        z_prev = z_t

    sequence_z = np.concatenate(sequence_z, axis=1)
    sequence_output = np.concatenate(sequence_output, axis=1)
    # print(sequence_output.shape)

    # n_plots = 5
    # fig, axes = plt.subplots(nrows=n_plots, ncols=1)
    # x = range(sequence_output.shape[1])
    # for i in range(n_plots):
    #     input = mini_batch[i, :].numpy().squeeze()
    #     output = sequence_output[i, :]
    #     axes[i].plot(range(input.shape[0]), input)
    #     axes[i].plot(range(len(output)), output)
    #     axes[i].grid()

    return mini_batch, sequence_z, sequence_output#fig


# helper function for doing evaluation
def do_evaluation():
    # put the RNN into evaluation mode (i.e. turn off drop-out if applicable)
    dmm.rnn.eval()

    # compute the validation and test loss n_samples many times
    val_nll = svi.evaluate_loss(val_batch, val_batch_reversed, val_batch_mask,
                                val_seq_lengths) / torch.sum(val_seq_lengths)
    test_nll = svi.evaluate_loss(test_batch, test_batch_reversed, test_batch_mask,
                                 test_seq_lengths) / torch.sum(test_seq_lengths)

    # put the RNN back into training mode (i.e. turn on drop-out if applicable)
    dmm.rnn.train()
    return val_nll, test_nll


# setup, training, and evaluation
def main(args):
    # setup logging
    log = get_logger(args.log)
    log(args)

    root_dir = r'D:\projects\trading\mlbootcamp\tickers'
    series_length = 60
    lookback = 50#160
    input_dim = 1
    train_start_date = datetime.date(2010, 1, 1)
    train_end_date = datetime.date(2016, 1, 1)
    test_start_date = train_end_date
    test_end_date = datetime.date(2019, 1, 1)
    min_sequence_length_train = 2 * (series_length + lookback)
    min_sequence_length_test = 2 * (series_length + lookback)

    dataset_train = create_ticker_dataset(root_dir, series_length, lookback, min_sequence_length_train, start_date=train_start_date, end_date=train_end_date)
    dataset_test = create_ticker_dataset(root_dir, series_length, lookback, min_sequence_length_test, start_date=test_start_date, end_date=test_end_date)
    dataloader_train = DataLoader(dataset_train, batch_size=args.mini_batch_size, shuffle=True, num_workers=0, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.mini_batch_size, shuffle=False, num_workers=0, drop_last=True)

    N_train_data = len(dataset_train)
    N_test_data = len(dataset_test)
    N_mini_batches = N_train_data // args.mini_batch_size
    N_train_time_slices = args.mini_batch_size * N_mini_batches

    print(f'N_train_data: {N_train_data}, N_test_data: {N_test_data}')

    # how often we do validation/test evaluation during training
    val_test_frequency = 50
    # the number of samples we use to do the evaluation
    n_eval_samples = 1

    # instantiate the dmm
    dmm = DMM(input_dim=input_dim, rnn_dropout_rate=args.rnn_dropout_rate, num_iafs=args.num_iafs,
              iaf_dim=args.iaf_dim, use_cuda=args.cuda)

    # setup optimizer
    adam_params = {"lr": args.learning_rate, "betas": (args.beta1, args.beta2),
                   "clip_norm": args.clip_norm, "lrd": args.lr_decay,
                   "weight_decay": args.weight_decay}
    adam = ClippedAdam(adam_params)

    # setup inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(dmm.model, dmm.guide, adam, loss=elbo)

    # if checkpoint files provided, load model and optimizer states from disk before we start training
    if args.load_opt != '' and args.load_model != '':
        load_checkpoint(dmm, adam, log)

    #################
    # TRAINING LOOP #
    #################
    times = [time.time()]
    for epoch in range(args.num_epochs):
        print(f'Starting epoch {epoch}.')
        # accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
        epoch_nll = 0.0
        # prepare mini-batch subsampling indices for this epoch
        shuffled_indices = torch.randperm(N_train_data)

        # process each mini-batch; this is where we take gradient steps
        dmm.train()
        for which_mini_batch in range(N_mini_batches):
            print(f'Epoch {epoch} of {args.num_epochs}, Batch {which_mini_batch} of {N_mini_batches}.')
            mini_batch = next(iter(dataloader_train))
            epoch_nll += process_minibatch(svi, epoch, mini_batch, N_mini_batches, which_mini_batch, shuffled_indices)

        # if specified, save model and optimizer states to disk every checkpoint_freq epochs
        if 1:#args.checkpoint_freq > 0 and epoch > 0 and epoch % args.checkpoint_freq == 0:
            save_checkpoint(dmm, adam, log)

        # report training diagnostics
        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        log("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" %
            (epoch, epoch_nll / N_train_time_slices, epoch_time))

        # do evaluation on test and validation data and report results
        if val_test_frequency > 0 and epoch > 0 and epoch % val_test_frequency == 0:
            val_nll, test_nll = do_evaluation()
            log("[val/test epoch %04d]  %.4f  %.4f" % (epoch, val_nll, test_nll))

        # Testing.
        print(f"Testing epoch {epoch}.")
        dmm.eval()
        mini_batch = next(iter(dataloader_test))
        fig = test_minibatch(dmm, mini_batch, args)
        fig.savefig(f'test_batch_{epoch}.png')
        plt.close(fig)

        # if 1:
        #     fig, _, _ = run_tsne(dmm, dataloader_test)
        #     fig.savefig(f'tsne_{epoch}.png')
        #     plt.close(fig)

    print("Testing")
    if 1:
        dmm.eval()
        mini_batch = next(iter(dataloader_test))

        x, z, x_reconst = test_minibatch(dmm, mini_batch, args)

        n_plots = 5
        fig, axes = plt.subplots(nrows=n_plots, ncols=1)
        for i in range(n_plots):
            input = x[i, :].numpy().squeeze()
            output = x_reconst[i, :]
            axes[i].plot(range(input.shape[0]), input)
            axes[i].plot(range(len(output)), output)
            axes[i].grid()
        fig.savefig(f'test_batch.png')
        plt.close(fig)

    if 1:
        # t-SNE.
        all_z_latents = []
        for test_batch in dataloader_test:
            # z_latents = minibatch_inference(dmm, test_batch)
            # z_latents = encode_x_to_z(dmm, test_batch, sample_z_t=False)
            x, z, x_reconst = test_minibatch(dmm, test_batch, args, sample_z=True)

            all_z_latents.append(z[:, x.shape[1] - 1, :])
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

        fig.savefig(f'tsne_test.png')
        plt.close(fig)

    return

    # Show some samples surrounding a given point.
    mini_batch = next(iter(dataloader_test))
    # Use the inference network to determine the parameters of the latents.
    z_loc, z_scale = minibatch_latent_parameters(dmm, mini_batch)
    z = sample_latent_sequence(dmm, z_loc, z_scale)

    most_recent_latents = latents[:, -1, :]
    # Take
    n_random_samples = 9


    print('Finished')


# parse command-line arguments and execute the main method
if __name__ == '__main__':
    assert pyro.__version__.startswith('0.4.1')

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', type=int, default=0) # 5000
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0003)
    parser.add_argument('-b1', '--beta1', type=float, default=0.96)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip-norm', type=float, default=20.0)
    parser.add_argument('-lrd', '--lr-decay', type=float, default=0.99996)
    parser.add_argument('-wd', '--weight-decay', type=float, default=2.0)
    parser.add_argument('-mbs', '--mini-batch-size', type=int, default=20)
    parser.add_argument('-ae', '--annealing-epochs', type=int, default=1000)
    parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.1)
    parser.add_argument('-rdr', '--rnn-dropout-rate', type=float, default=0.1)
    parser.add_argument('-iafs', '--num-iafs', type=int, default=0)
    parser.add_argument('-id', '--iaf-dim', type=int, default=100)
    parser.add_argument('-cf', '--checkpoint-freq', type=int, default=1)
    parser.add_argument('-lopt', '--load-opt', type=str, default='saved_opt_returns.pt')#saved_opt.pt')
    parser.add_argument('-lmod', '--load-model', type=str, default='saved_model_returns.pt')#saved_model.pt')
    parser.add_argument('-sopt', '--save-opt', type=str, default='saved_opt_returns.pt')
    parser.add_argument('-smod', '--save-model', type=str, default='saved_model_returns.pt')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('-l', '--log', type=str, default='dmm.log')
    args = parser.parse_args()

    main(args)
