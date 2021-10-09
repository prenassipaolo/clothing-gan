import torch


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
    pxz_distr = torch.distributions.Normal(x_hat, scale)

    # measure log-prob of seeing image under p(x|z)
    log_pxz = pxz_distr.log_prob(x)

    recon_loss = log_pxz.sum(dim=(1, 2, 3))

    return recon_loss


def elbo_loss(x, mu, sigma, z, x_hat, log_scale):

    # kl
    kl = kl_divergence(mu, sigma, z)

    # reconstruction loss
    recon_loss = reconstruction_loss(x, x_hat, log_scale)

    # elbo_loss average on batch
    elbo = (kl - recon_loss).mean()

    return elbo
