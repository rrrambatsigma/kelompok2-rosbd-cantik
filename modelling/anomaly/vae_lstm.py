import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

FEATURE_NAMES = [
    "latitude", "longitude", "velocity",
    "baro_altitude", "true_track",
    "dlat", "dlon", "dvel", "dalt", "dtrack"
]
N_FEATURES = len(FEATURE_NAMES)  # 10 (5 original + 5 derived deltas)
WINDOW_SIZE = 10


class VAELSTM(nn.Module):
    """
    VAE dengan encoder dan decoder LSTM untuk data sequence.

    Arsitektur:
      Encoder:  LSTM(5 → 64) → fc_mu(16), fc_logvar(16)
      Decoder:  fc(16→64) → repeat(10) → LSTM(64→64) → fc(64→5)
    """
    def __init__(self, n_features=N_FEATURES, window_size=WINDOW_SIZE,
                 hidden_dim=64, latent_dim=16):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # ── Encoder ──
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ── Decoder ──
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, n_features)

    def encode(self, x):
        """
        Encode input sequence ke mu dan logvar.

        Args:
            x: tensor (batch, window_size, n_features)
        Returns:
            mu:     tensor (batch, latent_dim)
            logvar: tensor (batch, latent_dim)
        """
        # LSTM output: (batch, window_size, hidden_dim)
        # Ambil hidden state terakhir (output[:, -1, :])
        lstm_out, _ = self.encoder_lstm(x)
        h = lstm_out[:, -1, :]          # (batch, hidden_dim)
        mu = self.fc_mu(h)              # (batch, latent_dim)
        logvar = self.fc_logvar(h)      # (batch, latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sampling dari distribusi normal: z = mu + eps * sigma."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode latent z kembali ke sequence.

        Args:
            z: tensor (batch, latent_dim)
        Returns:
            recon: tensor (batch, window_size, n_features)
        """
        # Expand ke hidden_dim
        h = self.fc_dec(z)              # (batch, hidden_dim)

        # Repeat untuk setiap timestep dalam window
        h = h.unsqueeze(1).repeat(1, self.window_size, 1)  # (batch, 10, hidden_dim)

        # LSTM decoder
        lstm_out, _ = self.decoder_lstm(h)  # (batch, 10, hidden_dim)

        # Proyeksi ke fitur asli
        recon = self.fc_out(lstm_out)       # (batch, 10, n_features)
        return recon

    def forward(self, x):
        """
        Forward pass lengkap.

        Args:
            x: tensor (batch, window_size, n_features)
        Returns:
            recon:  tensor (batch, window_size, n_features)
            mu:     tensor (batch, latent_dim)
            logvar: tensor (batch, latent_dim)
            z:      tensor (batch, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def vae_loss(recon_x, x, mu, logvar, beta=0.001):
    """
    Loss function VAE:
      L = Reconstruction MSE + beta * KL Divergence

    Args:
        recon_x: tensor (batch, window_size, n_features)
        x:       tensor (batch, window_size, n_features)
        mu:      tensor (batch, latent_dim)
        logvar:  tensor (batch, latent_dim)
        beta:    float — weight untuk KL divergence
    Returns:
        loss:       scalar tensor
        recon_loss: scalar tensor
        kl_loss:    scalar tensor
    """
    # Reconstruction loss: MSE antara input dan output
    recon_loss = nn.MSELoss(reduction='mean')(recon_x, x)

    # KL Divergence: ukur seberapa jauh distribusi latent dari N(0,1)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


class EarlyStopping:
    """Early stopping — hentikan training jika loss val tidak turun."""
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
