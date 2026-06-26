import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

FEATURE_NAMES = [
    "longitude", "latitude", "velocity",
    "geo_altitude", "true_track", "vertical_rate"
]


class VAE(nn.Module):
    def __init__(self, input_dim=6, latent_dim=4, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16, 8]

        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)

        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def vae_loss(recon_x, x, mu, logvar, beta=0.001):
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


class VAESVDD:
    def __init__(self, input_dim=6, latent_dim=4, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.vae = VAE(input_dim, latent_dim).to(self.device)
        self.svdd = None
        self.scaler = StandardScaler()
        self.recon_error_threshold = None
        self.svdd_threshold = None
        self.feature_names = FEATURE_NAMES[:]

    def fit_scaler(self, X):
        self.scaler.fit(X)

    def train_vae(self, X, epochs=200, batch_size=256, lr=1e-3, beta=0.001, verbose=True):
        X_scaled = self.scaler.transform(X)
        dataset = TensorDataset(torch.FloatTensor(X_scaled).to(self.device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.vae.parameters(), lr=lr)
        self.vae.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0]
                recon, mu, logvar, _ = self.vae(x)
                loss = vae_loss(recon, x, mu, logvar, beta)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if verbose and (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(loader.dataset)
                print(f"  VAE Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def extract_latent(self, X):
        self.vae.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            tensor = torch.FloatTensor(X_scaled).to(self.device)
            _, _, _, z = self.vae(tensor)
            return z.cpu().numpy()

    def train_svdd(self, X, nu=0.05, gamma='auto'):
        latent = self.extract_latent(X)
        self.svdd = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
        self.svdd.fit(latent)

    def compute_thresholds(self, X, quantile=0.95):
        self.vae.eval()
        X_scaled = self.scaler.transform(X)
        tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            recon, _, _, z = self.vae(tensor)
            recon_error = torch.mean((tensor - recon) ** 2, dim=1).cpu().numpy()

        latent = z.cpu().numpy()
        svdd_scores = self.svdd.decision_function(latent)
        svdd_dist = -svdd_scores

        self.recon_error_threshold = np.quantile(recon_error, quantile)
        self.svdd_threshold = np.quantile(svdd_dist, quantile)

    def predict(self, X):
        self.vae.eval()
        X_scaled = self.scaler.transform(X)
        tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            recon, _, _, z = self.vae(tensor)

        recon_np = recon.cpu().numpy()
        X_np = X_scaled
        z_np = z.cpu().numpy()

        per_feature_error = (X_np - recon_np) ** 2
        recon_error = np.mean(per_feature_error, axis=1)

        svdd_scores = self.svdd.decision_function(z_np)
        svdd_preds = self.svdd.predict(z_np)
        svdd_dist = -svdd_scores

        if self.recon_error_threshold is not None:
            recon_anomaly = recon_error > self.recon_error_threshold
        else:
            recon_anomaly = np.zeros_like(recon_error, dtype=bool)

        if self.svdd_threshold is not None:
            svdd_anomaly = svdd_dist > self.svdd_threshold
        else:
            svdd_anomaly = svdd_preds == -1

        is_anomaly = recon_anomaly | svdd_anomaly

        attack_type = self._classify_attack(
            per_feature_error, recon_anomaly, svdd_anomaly, z_np
        )

        max_feature_idx = np.argmax(per_feature_error, axis=1)
        dominant_feature = [self.feature_names[i] for i in max_feature_idx]

        return {
            "is_anomaly": is_anomaly.tolist(),
            "recon_error": recon_error.tolist(),
            "svdd_score": svdd_scores.tolist(),
            "svdd_distance": svdd_dist.tolist(),
            "combined_score": (recon_error + svdd_dist).tolist(),
            "dominant_feature": dominant_feature,
            "attack_type": attack_type,
            "per_feature_error": per_feature_error.tolist(),
            "latent_z": z_np.tolist(),
        }

    def _classify_attack(self, per_feature_error, recon_anomaly, svdd_anomaly, z_np):
        results = []
        for i in range(len(per_feature_error)):
            if not recon_anomaly[i] and not svdd_anomaly[i]:
                results.append("normal")
                continue

            fe = per_feature_error[i]
            lat_lon_error = fe[0] + fe[1]
            vel_error = fe[2]
            alt_error = fe[3]
            track_error = fe[4]
            total = fe.sum() + 1e-10

            lat_lon_ratio = lat_lon_error / total
            vel_ratio = vel_error / total
            track_ratio = track_error / total

            if track_ratio > 0.5:
                results.append("heading_manipulation")
            elif vel_ratio > 0.5:
                results.append("velocity_drift")
            elif lat_lon_ratio > 0.5 and (fe[0] / total) < 0.4:
                results.append("random_position")
            elif lat_lon_ratio > 0.5:
                results.append("constant_position")
            elif total > 0.8:
                results.append("flight_merge")
            else:
                results.append("dos_deletion")

        return results

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "model_state_dict": self.vae.state_dict(),
        }, os.path.join(path, "vae.pt"))
        joblib.dump(self.svdd, os.path.join(path, "svdd.joblib"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.joblib"))
        config = {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "recon_error_threshold": float(self.recon_error_threshold) if self.recon_error_threshold is not None else None,
            "svdd_threshold": float(self.svdd_threshold) if self.svdd_threshold is not None else None,
            "feature_names": self.feature_names,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def load(self, path):
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            self.input_dim = config["input_dim"]
            self.latent_dim = config["latent_dim"]
            self.recon_error_threshold = config.get("recon_error_threshold")
            self.svdd_threshold = config.get("svdd_threshold")
            self.feature_names = config.get("feature_names", FEATURE_NAMES)

        self.vae = VAE(self.input_dim, self.latent_dim).to(self.device)
        checkpoint = torch.load(os.path.join(path, "vae.pt"), map_location=self.device)
        self.vae.load_state_dict(checkpoint["model_state_dict"])
        self.vae.eval()

        self.svdd = joblib.load(os.path.join(path, "svdd.joblib"))
        self.scaler = joblib.load(os.path.join(path, "scaler.joblib"))
