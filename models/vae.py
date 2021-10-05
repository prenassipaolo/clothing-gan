import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim : int = 256, hidden_dims: list = [32, 64, 128, 256], im_dim: int = 32, feedforward_block_dim: int = 256):
        super().__init__()

        self.mu = None
        self.log_var = None

        modules = []

        for hidden_dim in hidden_dims:
            # downscaling factor equal to 2
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=hidden_dim,
                        kernel_size= 3,
                        stride= 2,
                        padding  = 1
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*modules)

        encoder_output_H_dim = im_dim // 2 ** len(hidden_dims)
        encoder_output_CxHxW_dim = hidden_dims[-1]*(encoder_output_H_dim ** 2)

        self.feedforward_block = nn.Sequential(
                                    nn.Linear(encoder_output_CxHxW_dim, feedforward_block_dim),
                                    nn.BatchNorm1d(feedforward_block_dim),
                                    nn.LeakyReLU()
                                 )

        self.feedforward_mu = nn.Linear(feedforward_block_dim, latent_dim)
        self.feedforward_sigma = nn.Linear(feedforward_block_dim, latent_dim)


    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.feedforward_block(x)

        self.mu = self.feedforward_mu(x)
        self.log_var = self.feedforward_sigma(x)

        return self.mu, self.log_var


class Sampling():
    def __init__(self):
        self.sigma = None
        self.z = None

    def __call__(self, mu, log_var):
        epsilon = torch.randn_like(log_var) # with dimension: batch * latent_dim
        epsilon = epsilon.type_as(mu) # Setting epsilon to be .cuda when using GPU training 
        self.sigma = torch.exp(0.5 * log_var)
        
        self.z = mu + epsilon * self.sigma

        return self.z


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3, latent_dim : int = 256, hidden_dims: list = [256, 128, 64, 32], im_dim: int = 32, feedforward_block_dim: int = 1024):
        super().__init__()

        self.feedforward_block_dim  = feedforward_block_dim
        self.dim_channel_0 = hidden_dims[0]

        # feedforward block
        self.feedforward_block = nn.Sequential(
            nn.Linear(latent_dim, feedforward_block_dim),
            nn.BatchNorm1d(feedforward_block_dim)
        )

        # deconvolution blocks
        modules = []

        for i in range(len(hidden_dims) - 1):
            # upscaling factor equal to 2
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_dims[i],
                        out_channels=hidden_dims[i + 1],
                        kernel_size= 3,
                        stride= 2,
                        padding  = 1,
                        output_padding=1    # necessary to adjust the dimensions: it adds zeros on the bottom right of the output
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        
        self.decoder = nn.Sequential(*modules)

        # last upscale to obtain the image dimension
        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_dims[-1],
                out_channels=hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=hidden_dims[-1], 
                out_channels= out_channels,
                kernel_size= 3, 
                padding= 1
            ),
            nn.Tanh()
        )


    def forward(self, z):
        z = self.feedforward_block(z)
        H_dim = int(np.sqrt(self.feedforward_block_dim/self.dim_channel_0))
        z = z.view(-1, self.dim_channel_0, H_dim, H_dim)
        x = self.decoder(z)
        x = self.final_block(x)

        return x


class VAE(nn.Module):
    def __init__(self, encoder, sampling, decoder):
        super().__init__()

        self.encoder = encoder
        self.sampling = sampling
        self.decoder = decoder

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        output = self.decoder(z)

        return output



def kl_divergence(mu, sigma, z):

    # create distributions p(z)~N(0,1), q(z|x)~N(mu,sigma^2)
    pz_distr = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
    qzx_distr = torch.distributions.Normal(mu, sigma)

    # get the log-probabilities of p(z) and q(z|x)
    log_pz = pz_distr.log_prob(z)
    log_qzx = qzx_distr.log_prob(z)
    
    # kl-diverence
    kl = (log_qzx - log_pz).sum(-1)

    return kl

def reconstruction_loss(x, x_hat, log_scale):

    # create distribution p(x|z)
    scale = torch.exp(log_scale) # learnable parameter
    mean = x_hat
    pxz_distr = torch.distributions.Normal(mean, scale)

    # measure log-prob of seeing image under p(x|z)
    log_pxz = pxz_distr.log_prob(x)

    recon_loss = log_pxz.sum(dim=(1, 2, 3))

    return recon_loss


def elbo_loss(mu, sigma, z, x, x_hat, log_scale):

    # kl
    kl = kl_divergence(mu, sigma, z)

    # reconstruction loss
    recon_loss = reconstruction_loss(x, x_hat, log_scale)

    # elbo_loss average on batch
    elbo = (kl - recon_loss).mean()

    return elbo

'''
### EXAMPLE

from torchsummary import summary

x = torch.ones([20,3,32,32])
enc = Encoder(in_channels=3, latent_dim= 256, hidden_dims=[32, 64, 128, 256], im_dim = 32, feedforward_block_dim = 256)
sampl = Sampling()
dec = Decoder()

vae = VAE(enc, sampl, dec)

x_hat = vae(x)

elbo = elbo_loss(
    vae.encoder.mu, 
    vae.sampling.sigma, 
    vae.sampling.z, 
    x, 
    x_hat, 
    torch.tensor([0]))

print(elbo)
#print(sampl)

#print(summary(vae))
'''
