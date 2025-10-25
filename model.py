import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, activation_fn=None):
        super(VAE, self).__init__()
        
        # Default to ReLU if no activation function is provided
        if activation_fn is None:
            activation_fn = nn.ReLU()
            
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Added sigmoid for BCE loss
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        recon_x = self.decoder(z)
        return recon_x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var


def loss_function(recon_x, x, mu, log_var, mask, lambda_weight, kl_weight=0.0):
    """MSE reconstruction loss + KL divergence loss"""
    recon_loss = nn.MSELoss(reduction='none')(recon_x, x)
    recon_loss = (recon_loss * (1 - mask) * (1 - lambda_weight) + recon_loss * mask * lambda_weight).sum(dim=-1).mean()
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
    return recon_loss + kl_loss * kl_weight


def bce_kl_loss(recon_x, x, mu, log_var, mask, lambda_weight, kl_weight=0.0):
    """BCE reconstruction loss + KL divergence loss"""
    recon_loss = nn.BCELoss(reduction='none')(recon_x, x)
    recon_loss = (recon_loss * (1 - mask) * (1 - lambda_weight) + recon_loss * mask * lambda_weight).sum(dim=-1).mean()
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
    return recon_loss + kl_loss * kl_weight