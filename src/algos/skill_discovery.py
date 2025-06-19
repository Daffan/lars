import torch
import torch.nn as nn
from torch import distributions as tdist


class Encoder(nn.Module):
    """ Encode Q or (Q, SA) to z, they have the same input dim
    """
    def __init__(self, input_dim, hidden_dim, hidden_layer, latent_dim):
        super(Encoder, self).__init__()

        self.fc = []
        for _ in range(hidden_layer):
            self.fc.append(nn.Linear(input_dim, hidden_dim))
            self.fc.append(nn.LeakyReLU(0.2))
            input_dim = hidden_dim
        self.fc = nn.Sequential(*self.fc)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear(hidden_dim, latent_dim)
        self.training = True
        
    def forward(self, x):
        h_       = self.fc(x)
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        return mean, log_var
        

class Decoder(nn.Module):
    """ Decode (Q, z) to SA
    """
    def __init__(self, input_dim, latent_dim, hidden_dim, hidden_layer, output_dim):
        super(Decoder, self).__init__()
        self.fc = []
        input_dim += latent_dim
        for _ in range(hidden_layer):
            self.fc.append(nn.Linear(input_dim, hidden_dim))
            self.fc.append(nn.LeakyReLU(0.2))
            input_dim = hidden_dim
        self.fc = nn.Sequential(*self.fc)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, q, z):
        x = torch.cat([q, z], dim=1)
        h = self.fc(x)
        qsa_hat = self.FC_output(h)
        return qsa_hat
    

class Model(nn.Module):
    def __init__(self, q_encoder, qs_encoder, pi, decoder, device):
        super(Model, self).__init__()
        self.q_encoder = q_encoder
        self.qs_encoder = qs_encoder
        self.pi = pi
        self.decoder = decoder
        self.device = device

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)        # sampling epsilon        
        z = mean + var * epsilon                          # reparameterization trick
        return z


def loss_function(x, x_hat, prior_mean, prior_log_var, post_mean, post_log_var):
    reconstruction_loss = torch.mean(torch.square(x_hat - x, dim=1), dim=1)
    # this is computed analytically
    prior_dist = tdist.Normal(prior_mean, torch.exp(prior_log_var))
    post_dist = tdist.Normal(post_mean, torch.exp(post_log_var))
    KLD = -0.5 * tdist.kl_divergence(post_dist, prior_dist)

    return reconstruction_loss + KLD

def get_batched_loss(q_batch, s_batch, qs_batch, y_batch, kl_ratio, model: Model):
    prior_mean, prior_log_var = model.q_encoder(q_batch)
    post_mean, post_log_var = model.qs_encoder(qs_batch)

    prior_dist = tdist.Normal(prior_mean, torch.exp(prior_log_var))
    post_dist = tdist.Normal(post_mean, torch.exp(post_log_var))

    z_batch = model.reparameterization(post_mean, post_log_var)
    s_hat_batch = model.decoder(q_batch, z_batch)

    # reconstruction_loss = torch.mean(torch.sum(torch.square(sa_hat_batch - sa_batch), dim=1))
    reconstruction_loss = torch.mean(torch.sum(torch.square(s_hat_batch - s_batch), dim=1))
    # Compute KL analytically
    kld = kl_ratio * torch.mean(torch.sum(tdist.kl_divergence(post_dist, prior_dist), dim=1))
    
    # policy loss
    with torch.no_grad():
        post_mean, _ = model.qs_encoder(qs_batch)
    pi_mean, pi_log_var = model.pi(q_batch)
    pi_dist = tdist.Normal(pi_mean, torch.exp(pi_log_var))
    pi_loss = -torch.sum(pi_dist.log_prob(post_mean), dim=1) * y_batch * 0.01 # * (y_batch - 0.5) * 0.01
    pi_loss = torch.mean(pi_loss)

    return reconstruction_loss + kld + pi_loss, (reconstruction_loss, kld, pi_loss)