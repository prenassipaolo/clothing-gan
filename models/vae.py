import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim : int = 256, hidden_dims: list = [32, 64, 128, 256], im_dim: int = 32, feedforward_block_dim: int = 20):
        super().__init__()

        modules = []
        for hidden_dim in hidden_dims:
            # downscaling factor equal to 2
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=hidden_dim,
                              kernel_size= 3,
                              stride= 2,
                              padding  = 1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU())
            )
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*modules)

        encoder_output_H_dim = im_dim / 2 ** len(hidden_dims)
        encoder_output_chxHxW_dim = hidden_dims[-1]*(encoder_output_H_dim ** 2)

        self.feedforward_block = nn.Sequential(nn.Linear(encoder_output_chxHxW_dim, feedforward_block_dim),
                                               nn.BatchNorm2d(feedforward_block_dim),
                                               nn.LeakyReLU()
                                               )

        self.feedforward_mu = nn.Linear(feedforward_block_dim, latent_dim)
        self.feedforward_sigma = nn.Linear(feedforward_block_dim, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.feedforward_block(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.feedforward_mu(x)
        log_sigma = self.feedforward_sigma(x)

        return mu, log_sigma


# lass Sampling(nn.Module):

# class Decoder(nn.Module)

# class VAE(nn.Module):
