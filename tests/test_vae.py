import unittest
import torch
from src.models.vae import VAE, vae_loss_function

class TestVAE(unittest.TestCase):
    def setUp(self):
        self.input_dim = 512
        self.latent_dim = 128
        self.batch_size = 8
        self.vae = VAE(self.input_dim, self.latent_dim)

    def test_forward_shapes(self):
        """Verify that input is compressed and then reconstructed to original size."""
        x = torch.randn((self.batch_size, self.input_dim))
        recon, mu, logvar = self.vae(x)

        self.assertEqual(recon.shape, (self.batch_size, self.input_dim), "Reconstruction shape mismatch")
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim), "Mu shape mismatch")
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim), "Logvar shape mismatch")

    def test_loss_calculation(self):
        """Verify that the loss function returns a finite positive value."""
        x = torch.randn((self.batch_size, self.input_dim))
        recon, mu, logvar = self.vae(x)
        loss = vae_loss_function(recon, x, mu, logvar)

        self.assertTrue(torch.isfinite(loss), "VAE loss is NaN or Inf")
        self.assertGreater(loss.item(), 0, "VAE loss should be positive")

    def test_reparameterization(self):
        """Ensure that the reparameterization trick produces different samples for same mu/logvar."""
        mu = torch.zeros((1, self.latent_dim))
        logvar = torch.zeros((1, self.latent_dim))

        z1 = self.vae.reparameterize(mu, logvar)
        z2 = self.vae.reparameterize(mu, logvar)

        self.assertFalse(torch.equal(z1, z2), "Reparameterization should be stochastic")

if __name__ == "__main__":
    unittest.main()
