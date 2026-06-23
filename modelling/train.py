"""
train.py — Tahap 2: Training VAE-LSTM + OneClassSVM

ALUR:
  1. Load data .npy dari data/checkpoint/
  2. Fit StandardScaler di TRAIN SAJA (jangan bocor!)
  3. Train VAE-LSTM
  4. Extract latent z dari encoder (pakai mu, tanpa sampling)
  5. Train OneClassSVM di latent z
  6. Hitung threshold dari VAL set (persentil 95 combined score)
  7. Evaluasi di test set (classification report + AUC-ROC)
  8. Simpan artifacts ke models/vae-svdd/

PRINSIP:
  - Scaler.fit() HANYA dari train data
  - Val set untuk threshold (100% normal)
  - Test set untuk evaluasi (26% anomali, 6 jenis)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from modelling.vae_lstm import VAELSTM, vae_loss, EarlyStopping, FEATURE_NAMES

# ─── KONFIGURASI ────────────────────────────────────────────
DATA_DIR = os.path.join("data", "checkpoint")
MODEL_DIR = os.path.join("models", "vae-svdd")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── LOGGING ────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


# ─── LANGKAH 1: LOAD DATA ─────────────────────────────────
def load_data(data_dir: str):
    """Load .npy files dari data/checkpoint/."""
    train = np.load(os.path.join(data_dir, "train_data.npy")).astype(np.float32)
    val   = np.load(os.path.join(data_dir, "val_data.npy")).astype(np.float32)
    test  = np.load(os.path.join(data_dir, "test_data.npy")).astype(np.float32)
    labels = np.load(os.path.join(data_dir, "test_labels.npy"))

    log(f"Load data:")
    log(f"  Train: {train.shape} — 100% normal ({train.shape[0]} windows)")
    log(f"  Val:   {val.shape}   — 100% normal ({val.shape[0]} windows)")
    log(f"  Test:  {test.shape}  — {labels.mean()*100:.1f}% anomaly ({test.shape[0]} windows)")
    return train, val, test, labels


# ─── LANGKAH 2: SCALER ─────────────────────────────────────
def fit_scaler(train_data: np.ndarray) -> StandardScaler:
    """
    Fit StandardScaler dari TRAIN DATA SAJA.
    Reshape: (N, 10, 5) -> (N*10, 5) untuk fit -> transform.
    """
    N, T, F = train_data.shape
    train_2d = train_data.reshape(-1, F)

    scaler = StandardScaler()
    scaler.fit(train_2d)
    log(f"Scaler fitted on train: {len(train_2d)} records, {F} features")
    log(f"  Means:  {np.round(scaler.mean_, 2)}")
    log(f"  Stds:   {np.round(scaler.scale_, 2)}")
    return scaler


def scale_data(scaler: StandardScaler, data: np.ndarray) -> np.ndarray:
    """Scale data dengan scaler yang sudah di-fit."""
    N, T, F = data.shape
    data_2d = data.reshape(-1, F)
    scaled = scaler.transform(data_2d)
    return scaled.reshape(N, T, F)


# ─── LANGKAH 3: TRAIN VAE-LSTM ────────────────────────────
def train_vae(model, train_loader, val_loader, epochs, lr, beta, device, patience=20):
    """Training loop VAE-LSTM dengan early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stop = EarlyStopping(patience=patience)
    history = {"train_loss": [], "val_loss": [], "recon_loss": [], "kl_loss": []}

    log(f"Training VAE-LSTM pada device: {device}")
    log(f"  Epochs: {epochs}, LR: {lr}, Beta: {beta}, Patience: {patience}")

    for epoch in range(epochs):
        # ── Training ──
        model.train()
        train_total = 0
        train_recon = 0
        train_kl = 0
        for batch in train_loader:
            x = batch[0].to(device)
            recon, mu, logvar, _ = model(x)
            loss, r_loss, k_loss = vae_loss(recon, x, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total += loss.item()
            train_recon += r_loss.item()
            train_kl += k_loss.item()

        avg_train = train_total / len(train_loader)
        avg_recon = train_recon / len(train_loader)
        avg_kl = train_kl / len(train_loader)

        # ── Validation ──
        model.eval()
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                recon, mu, logvar, _ = model(x)
                loss, _, _ = vae_loss(recon, x, mu, logvar, beta)
                val_total += loss.item()
        avg_val = val_total / len(val_loader)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            log(f"  Epoch {epoch+1:3d}/{epochs}: "
                f"train={avg_train:.4f} val={avg_val:.4f} "
                f"(recon={avg_recon:.4f} kl={avg_kl:.4f})")

        # Early stopping
        if early_stop.step(avg_val, model):
            log(f"  Early stopping di epoch {epoch+1}")
            break

    # Restore best model
    early_stop.restore(model)
    log(f"  Best val loss: {early_stop.best_loss:.4f}")
    return history


# ─── LANGKAH 4-5: LATENT + SVDD ────────────────────────────
def extract_latent(model, data, scaler, device):
    """
    Extract latent representation z (mu, tanpa sampling).

    Args:
        data: numpy array (N, 10, 5) — sudah di-scale
    Returns:
        z: numpy array (N, latent_dim)
    """
    model.eval()
    tensor = torch.FloatTensor(data).to(device)
    with torch.no_grad():
        mu, _ = model.encode(tensor)
    return mu.cpu().numpy()


def compute_scores(model, data, scaler, svdd, device):
    """
    Hitung reconstruction error + SVDD distance.

    Args:
        data: numpy array (N, 10, 5)
    Returns:
        dict dengan keys: recon_error, svdd_score, svdd_dist, combined
    """
    model.eval()
    tensor = torch.FloatTensor(data).to(device)

    with torch.no_grad():
        recon, mu, _, _ = model(tensor)

    recon_np = recon.cpu().numpy()
    data_np = data
    z_np = mu.cpu().numpy()

    # Reconstruction error: MSE per window
    recon_error = np.mean((data_np - recon_np) ** 2, axis=(1, 2))

    # SVDD distance
    svdd_scores = svdd.decision_function(z_np)
    svdd_dist = -svdd_scores  # makin besar = makin anomali

    combined = recon_error + svdd_dist

    return {
        "recon_error": recon_error,
        "svdd_score": svdd_scores,
        "svdd_dist": svdd_dist,
        "combined": combined,
        "latent": z_np,
        "recon": recon_np,
    }


def classify_attack(recon, original, z, feature_names):
    """
    Tentukan jenis anomali berdasarkan per-feature reconstruction error.

    Args:
        recon: numpy (batch, 10, 5) — hasil rekonstruksi
        original: numpy (batch, 10, 5) — data asli
        z: numpy (batch, latent_dim) — latent
    Returns:
        list of attack type strings
    """
    per_feature_error = np.mean((original - recon) ** 2, axis=1)  # (batch, 5)
    results = []

    for i in range(len(per_feature_error)):
        fe = per_feature_error[i]
        total = fe.sum() + 1e-10

        lat_lon_error = fe[0] + fe[1]
        vel_error = fe[2]
        alt_error = fe[3]
        track_error = fe[4]

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


# ─── LANGKAH 6-7: THRESHOLD + EVALUASI ─────────────────────
def find_threshold(scores_val: np.ndarray, percentile: float = 95):
    """Tentukan threshold dari persentil val set."""
    threshold = np.percentile(scores_val, percentile)
    log(f"Threshold (P{percentile} dari val): {threshold:.4f}")
    return threshold


def evaluate(test_data, test_labels, model, scaler, svdd,
             threshold, recon_threshold, device):
    """Evaluasi model di test set."""
    scores = compute_scores(model, test_data, scaler, svdd, device)
    y_true = test_labels.astype(int)

    # Dua metode scoring: combined (VAE+SVDD) dan recon-only
    y_pred_combined = (scores["combined"] > threshold).astype(int)
    y_pred_recon = (scores["recon_error"] > recon_threshold).astype(int)

    # Cek distribusi skor per class
    log(f"\n  Score Distributions (test set):")
    for name, arr in [("recon_error", scores["recon_error"]),
                      ("svdd_dist", scores["svdd_dist"]),
                      ("combined", scores["combined"])]:
        log(f"    {name:15s}: min={arr.min():.3f} max={arr.max():.3f} "
            f"mean={arr.mean():.3f} median={np.median(arr):.3f} "
            f"p95={np.percentile(arr,95):.3f}")
    log(f"    threshold (combined): {threshold:.3f}")
    log(f"    threshold (recon):    {recon_threshold:.3f}")
    for name in ["normal", "anomaly"]:
        mask = y_true == (1 if name == "anomaly" else 0)
        if mask.sum() > 0:
            for score_name in ["recon_error", "svdd_dist", "combined"]:
                arr = scores[score_name][mask]
                log(f"    {name:10s} {score_name:15s}: "
                    f"mean={arr.mean():.3f} median={np.median(arr):.3f}")

    # Evaluasi combined
    log(f"\n  --- COMBINED (VAE + SVDD) ---")
    y_pred = y_pred_combined
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    auc = roc_auc_score(y_true, scores["combined"])

    log(f"  Confusion Matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
    log(f"  Accuracy:  {accuracy:.4f}  Precision: {precision:.4f}")
    log(f"  Recall:    {recall:.4f}  F1-Score:  {f1:.4f}")
    log(f"  AUC-ROC:   {auc:.4f}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    auc = roc_auc_score(y_true, scores["combined"])

    log(f"\n{'='*50}")
    log("EVALUASI TEST SET")
    log(f"{'='*50}")
    log(f"  Confusion Matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
    log(f"  Accuracy:  {accuracy:.4f}")
    log(f"  Precision: {precision:.4f}")
    log(f"  Recall:    {recall:.4f}")
    log(f"  F1-Score:  {f1:.4f}")
    log(f"  AUC-ROC:   {auc:.4f}")

    # Evaluasi reconstruction-only
    log(f"\n  --- RECONSTRUCTION ONLY ---")
    y_pred_r = y_pred_recon
    tn_r, fp_r, fn_r, tp_r = confusion_matrix(y_true, y_pred_r).ravel()
    auc_r = roc_auc_score(y_true, scores["recon_error"])
    log(f"  Confusion Matrix: TP={tp_r} FP={fp_r} FN={fn_r} TN={tn_r}")
    log(f"  AUC-ROC: {auc_r:.4f}")

    # Attack type classification (recon-only)
    log(f"\n  Per-Attack Detection (recon-only):")
    attack_types = np.load(os.path.join(DATA_DIR, "test_attack_types.npy"))
    for atk in np.unique(attack_types):
        if atk == "normal":
            continue
        mask = attack_types == atk
        n_total = mask.sum()
        n_detected = (y_pred_r[mask] == 1).sum()
        log(f"    {atk:25s}: {n_detected:3d}/{n_total:3d} detected "
            f"({100*n_detected/max(n_total,1):.0f}%)")

    return {
        "accuracy_combined": accuracy,
        "precision_combined": precision,
        "recall_combined": recall,
        "f1_combined": f1,
        "auc_combined": auc,
        "auc_recon": auc_r,
        "tp_combined": int(tp),
        "fp_combined": int(fp),
        "fn_combined": int(fn),
        "tn_combined": int(tn),
    }


# ─── LANGKAH 8: SIMPAN ─────────────────────────────────────
def save_artifacts(model, scaler, svdd, threshold, metrics,
                   model_dir: str, args):
    """Simpan model, scaler, dan konfigurasi."""
    os.makedirs(model_dir, exist_ok=True)

    # VAE model
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": model.n_features,
        "latent_dim": model.latent_dim,
        "window_size": model.window_size,
        "hidden_dim": model.hidden_dim,
    }, os.path.join(model_dir, "vae_model.pt"))
    log(f"  Saved: vae_model.pt")

    # SVDD
    joblib.dump(svdd, os.path.join(model_dir, "svdd_model.pkl"))
    log(f"  Saved: svdd_model.pkl")

    # Scaler
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    log(f"  Saved: scaler.pkl")

    # Config
    config = {
        "threshold": float(threshold),
        "feature_names": FEATURE_NAMES,
        "n_features": model.n_features,
        "latent_dim": model.latent_dim,
        "window_size": model.window_size,
        "hidden_dim": model.hidden_dim,
        "vae_epochs": args.epochs,
        "vae_lr": args.lr,
        "vae_beta": args.beta,
        "svdd_nu": args.nu,
        "svdd_gamma": args.gamma,
        "threshold_percentile": 95,
        "training_date": datetime.now().isoformat(),
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    log(f"  Saved: config.json")

    # Metrics
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"  Saved: metrics.json")

    log(f"\n  Semua artifacts tersimpan di: {model_dir}/")


# ─── MAIN ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train VAE-LSTM + OneClassSVM")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--gamma", type=str, default="auto")
    parser.add_argument("--threshold-percentile", type=float, default=95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log("=" * 60)
    log("TAHAP 2: TRAINING VAE-LSTM + OneClassSVM")
    log(f"Device: {DEVICE}")
    log("=" * 60)

    t_start = time.time()

    # ── Langkah 1: Load Data ──
    log("\n[1/8] Load data...")
    train_data, val_data, test_data, test_labels = load_data(args.data_dir)

    # ── Langkah 2: Scaler ──
    log("\n[2/8] Fit StandardScaler (dari TRAIN SAJA)...")
    scaler = fit_scaler(train_data)

    train_scaled = scale_data(scaler, train_data)
    val_scaled = scale_data(scaler, val_data)
    test_scaled = scale_data(scaler, test_data)

    # ── Langkah 3: Train VAE-LSTM ──
    log("\n[3/8] Train VAE-LSTM...")
    model = VAELSTM(
        n_features=train_data.shape[2],
        window_size=train_data.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(DEVICE)

    # DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(train_scaled))
    val_dataset = TensorDataset(torch.FloatTensor(val_scaled))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Training
    history = train_vae(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, beta=args.beta,
        device=DEVICE, patience=20
    )

    # ── Langkah 4-5: Latent + SVDD ──
    log("\n[4/8] Extract latent dari train set...")
    z_train = extract_latent(model, train_scaled, scaler, DEVICE)
    log(f"  Latent z shape: {z_train.shape}")

    log("\n[5/8] Train OneClassSVM...")
    svdd = OneClassSVM(kernel='rbf', gamma=args.gamma, nu=args.nu)
    svdd.fit(z_train)
    n_support = len(svdd.support_)
    log(f"  Support vectors: {n_support}/{len(z_train)} "
        f"({100*n_support/len(z_train):.1f}%)")

    # ── Langkah 6: Threshold ──
    log("\n[6/8] Compute threshold dari VAL set...")
    val_scores = compute_scores(model, val_scaled, scaler, svdd, DEVICE)

    # Combined threshold (VAE + SVDD)
    combined_threshold = find_threshold(val_scores["combined"], args.threshold_percentile)
    # Recon-only threshold (baseline)
    recon_threshold = find_threshold(val_scores["recon_error"], args.threshold_percentile)

    log(f"  Combined threshold: {combined_threshold:.4f}")
    log(f"  Recon threshold:    {recon_threshold:.4f}")
    log(f"  Recon error range: [{val_scores['recon_error'].min():.4f}, "
        f"{val_scores['recon_error'].max():.4f}]")
    log(f"  SVDD dist range:   [{val_scores['svdd_dist'].min():.4f}, "
        f"{val_scores['svdd_dist'].max():.4f}]")

    # ── Langkah 7: Evaluasi ──
    log("\n[7/8] Evaluasi di test set...")
    metrics = evaluate(test_scaled, test_labels, model, scaler, svdd,
                       combined_threshold, recon_threshold, DEVICE)

    # ── Langkah 8: Simpan ──
    log("\n[8/8] Simpan artifacts...")
    save_artifacts(model, scaler, svdd, combined_threshold, metrics,
                   args.model_dir, args)

    t_elapsed = time.time() - t_start
    log(f"\n{'=' * 60}")
    log(f"SELESAI! Waktu: {t_elapsed/60:.1f} menit")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
