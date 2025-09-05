import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        # (B, 1, 28, 28) → (B, 32, 14, 14)
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32,
            kernel_size=4, stride=2, padding=1)
        # (B, 32, 14, 14) → (B, 64, 7, 7)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64,
            kernel_size=4, stride=2, padding=1)
        self.fc_mu = nn.Linear(64*7*7, latent_dim)
        self.fc_logvar = nn.Linear(64*7*7, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128*7*7)
        # (B, 128, 7, 7) → (B, 64, 14, 14)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=4, stride=2, padding=1)
        # (B, 64, 14, 14) → (B, 32, 28, 28)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32,
            kernel_size=4, stride=2, padding=1)
        # (B, 32, 28, 28) → (B, 1, 28, 28)
        self.deconv3 = nn.Conv2d(
            in_channels=32, out_channels=1,
            kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
