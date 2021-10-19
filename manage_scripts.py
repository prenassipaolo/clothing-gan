import numpy as np


def example_vae_loss():

    from vae.vae import Encoder, Decoder, VAE
    from vae.loss import elbo_loss
    
    import torch
    from torchsummary import summary


    IN_CHANNELS, OUT_CHANNELS = 3, 3
    LATENT_DIM = 16#256
    HIDDEN_DIMS = [8, 16, 32]#[32, 64, 128, 256]
    IM_DIM = 32
    FEEDFORWARD_BLOCK_DIM_ENCODER = 64#1024
    decoder_conv_input_H_dim = IM_DIM // 2 ** len(HIDDEN_DIMS)
    decoder_conv_input_CxHxW_dim = HIDDEN_DIMS[-1]*(decoder_conv_input_H_dim ** 2)
    FEEDFORWARD_BLOCK_DIM_DECODER = decoder_conv_input_CxHxW_dim

    enc = Encoder(
        in_channels=IN_CHANNELS, 
        latent_dim=LATENT_DIM, 
        hidden_dims=HIDDEN_DIMS, 
        im_dim=IM_DIM, 
        feedforward_block_dim=FEEDFORWARD_BLOCK_DIM_ENCODER
        )
    dec = Decoder(
        out_channels=OUT_CHANNELS, 
        latent_dim=LATENT_DIM, 
        hidden_dims=HIDDEN_DIMS[::-1], # reverse the order of the hidden dimensions 
        im_dim=IM_DIM, 
        feedforward_block_dim=FEEDFORWARD_BLOCK_DIM_DECODER
        )
    vae = VAE(enc, dec)

    x = torch.ones([20,3,32,32])

    """
    enc = Encoder(in_channels=3, latent_dim= 256, hidden_dims=[32, 64, 128, 256], im_dim = 32, feedforward_block_dim = 256)
    enc = Encoder()
    dec = Decoder()

    vae = VAE(enc, dec)
    """

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

    from vae.train import Train

    T = Train()
    print(T.set_device())

    return



#example_vae_loss()
#example_train()

import matplotlib.pyplot as plt
from data.download_data import download_cifar_dataset, download_mnist_dataset
from vae.vae import Encoder, Decoder, VAE
from vae.train import Train


def train_vae():
    
    
    # MNIST
    train_set, test_set = download_mnist_dataset()
    IN_CHANNELS, OUT_CHANNELS = 1, 1
    LATENT_DIM = 16#256
    HIDDEN_DIMS = [8, 16, 32]
    IM_DIM = 32
    FEEDFORWARD_BLOCK_DIM_ENCODER = 64
    decoder_conv_input_H_dim = IM_DIM // 2 ** len(HIDDEN_DIMS)
    decoder_conv_input_CxHxW_dim = HIDDEN_DIMS[-1]*(decoder_conv_input_H_dim ** 2)
    FEEDFORWARD_BLOCK_DIM_DECODER = decoder_conv_input_CxHxW_dim
    '''
    # CIFAR
    train_set, test_set = download_cifar_dataset()
    IN_CHANNELS, OUT_CHANNELS = 3, 3
    LATENT_DIM = 16#256
    HIDDEN_DIMS = [8, 16, 32]#[32, 64, 128, 256]
    IM_DIM = 32
    FEEDFORWARD_BLOCK_DIM_ENCODER = 64#1024
    decoder_conv_input_H_dim = IM_DIM // 2 ** len(HIDDEN_DIMS)
    decoder_conv_input_CxHxW_dim = HIDDEN_DIMS[-1]*(decoder_conv_input_H_dim ** 2)
    FEEDFORWARD_BLOCK_DIM_DECODER = decoder_conv_input_CxHxW_dim
    '''

    enc = Encoder(
        in_channels=IN_CHANNELS, 
        latent_dim=LATENT_DIM, 
        hidden_dims=HIDDEN_DIMS, 
        im_dim=IM_DIM, 
        feedforward_block_dim=FEEDFORWARD_BLOCK_DIM_ENCODER
        )
    dec = Decoder(
        out_channels=OUT_CHANNELS, 
        latent_dim=LATENT_DIM, 
        hidden_dims=HIDDEN_DIMS[::-1], # reverse the order of the hidden dimensions 
        im_dim=IM_DIM, 
        feedforward_block_dim=FEEDFORWARD_BLOCK_DIM_DECODER
        )
    vae = VAE(enc, dec)

    train = Train(epochs=20, batch_size=128, log_interval=10)

    history = train(train_set, test_set, vae)

    print(history)

    plt.plot(history)
    plt.show()

    return enc, dec, vae, history

def plot_examples():

    from vae.vae import Encoder, Decoder, VAE
    import torch
    import matplotlib.pyplot as plt

    IN_CHANNELS, OUT_CHANNELS = 1, 1
    LATENT_DIM = 16#256
    HIDDEN_DIMS = [8, 16, 32]
    IM_DIM = 32
    FEEDFORWARD_BLOCK_DIM_ENCODER = 64
    decoder_conv_input_H_dim = IM_DIM // 2 ** len(HIDDEN_DIMS)
    decoder_conv_input_CxHxW_dim = HIDDEN_DIMS[-1]*(decoder_conv_input_H_dim ** 2)
    FEEDFORWARD_BLOCK_DIM_DECODER = decoder_conv_input_CxHxW_dim

    enc = Encoder(
        in_channels=IN_CHANNELS, 
        latent_dim=LATENT_DIM, 
        hidden_dims=HIDDEN_DIMS, 
        im_dim=IM_DIM, 
        feedforward_block_dim=FEEDFORWARD_BLOCK_DIM_ENCODER
        )
    dec = Decoder(
        out_channels=OUT_CHANNELS, 
        latent_dim=LATENT_DIM, 
        hidden_dims=HIDDEN_DIMS[::-1], # reverse the order of the hidden dimensions 
        im_dim=IM_DIM, 
        feedforward_block_dim=FEEDFORWARD_BLOCK_DIM_DECODER
        )
    vae = VAE(enc, dec)

    # load the model
    PATH = 'checkpoint/vae.pth'
    vae = VAE(enc, dec)
    vae.load_state_dict(torch.load(PATH))
    vae.eval()


    N_FAKE = 4
    z = torch.rand([N_FAKE,LATENT_DIM])
    fake_images = vae.decoder(z)

    imshow(fake_images)
    return fake_images





    return

def imshow(images):
    images = images / 2 + 0.5     # unnormalize
    #img = torch.squeeze(img)
    #npimg = img.detach().numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.imshow(npimg)

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(torch.squeeze(images[0]).detach().numpy())
    axarr[0,1].imshow(torch.squeeze(images[1]).detach().numpy())
    axarr[1,0].imshow(torch.squeeze(images[2]).detach().numpy())
    axarr[1,1].imshow(torch.squeeze(images[3]).detach().numpy())
    plt.show()
    return 




import torch

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    #enc, dec, vae, history = train_vae()
    #example_vae_loss()

    plot_examples()
