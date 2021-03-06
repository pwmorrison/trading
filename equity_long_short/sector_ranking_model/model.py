import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        # setup the three linear transformations used
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc11 = nn.Linear(hidden_dim, hidden_dim)
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
        hidden = self.softplus(self.fc11(hidden))
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
        # self.decoder = Decoder(z_dim, hidden_dim, x_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.x_dim = x_dim

    # define the model p(x|z)p(z)
    def model(self, x_1, x_2, y):
        # register PyTorch module `decoder` with Pyro
        # pyro.module("decoder", self.decoder)
        with pyro.plate("data", x_1.shape[0]):
            # setup hyperparameters for prior p(z)
            z_1_loc = torch.zeros(x_1.shape[0], self.z_dim, dtype=x_1.dtype, device=x_1.device)
            z_1_scale = torch.ones(x_1.shape[0], self.z_dim, dtype=x_1.dtype, device=x_1.device)
            z_2_loc = torch.zeros(x_2.shape[0], self.z_dim, dtype=x_2.dtype, device=x_2.device)
            z_2_scale = torch.ones(x_2.shape[0], self.z_dim, dtype=x_2.dtype, device=x_2.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z_1 = pyro.sample("latent_1", dist.Normal(z_1_loc, z_1_scale).to_event(1))
            z_2 = pyro.sample("latent_2", dist.Normal(z_2_loc, z_2_scale).to_event(1))
            # decode the latent code z
            if 0:
                loc_img = self.decoder.forward(z_1)
                # score against actual images
                # pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, self.x_dim))
                pyro.sample("obs", dist.Normal(loc_img, 0.1).to_event(1), obs=x_1.reshape(-1, self.x_dim))
                # return the loc so we can visualize it later
                return loc_img
            else:
                ret_sample = torch.sigmoid(z_1 - z_2)
                pyro.sample("obs", dist.Bernoulli(ret_sample).to_event(1), obs=y)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x_1, x_2, y):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data_1", x_1.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x_1)
            # sample the latent code z
            pyro.sample("latent_1", dist.Normal(z_loc, z_scale).to_event(1))
        with pyro.plate("data_2", x_2.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x_2)
            # sample the latent code z
            pyro.sample("latent_2", dist.Normal(z_loc, z_scale).to_event(1))

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
