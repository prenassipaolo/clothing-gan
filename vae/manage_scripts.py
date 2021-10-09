
def example_vae_loss():

    from vae import Encoder, Decoder, VAE
    from loss import elbo_loss
    
    import torch
    from torchsummary import summary

    x = torch.ones([20,3,32,32])
    enc = Encoder(in_channels=3, latent_dim= 256, hidden_dims=[32, 64, 128, 256], im_dim = 32, feedforward_block_dim = 256)
    dec = Decoder()

    vae = VAE(enc, dec)

    x_hat, mu, sigma, z = vae(x)

    elbo = elbo_loss(
        x,
        mu, 
        sigma, 
        z, 
        x_hat, 
        torch.tensor([0]))

    print(f'elbo: {elbo}\n')
    print(f'vae_summary:\n{summary(vae)}\n')

    return


def example_train():

    from train import Train

    T = Train()
    print(T.set_device())

    return



example_vae_loss()
#example_train()
