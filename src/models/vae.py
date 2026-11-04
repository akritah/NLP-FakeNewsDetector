import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for feature compression as described in the paper.
    The VAE compresses high-dimensional multimodal features into a latent space.
    """
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: Maps input to latent distribution parameters (mu and log_var)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder: Maps latent vector back to original input dimension
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and log-variance of the latent distribution.
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        The Reparameterization Trick: z = mu + std * epsilon
        Allows gradients to flow through the random sampling process.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the input from the latent vector z.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass.
        Returns: (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, kld_weight=1.0):
    """
    VAE Loss = Reconstruction Loss (MSE) + KL Divergence.
    """
    # 1. Reconstruction Loss: How well the VAE recovers the original features
    # Use MSE for continuous feature vectors
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # 2. KL Divergence: Forces the latent space to follow a standard Normal distribution N(0,1)
    # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + (kld_weight * kld_loss)
