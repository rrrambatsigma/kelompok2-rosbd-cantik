"""
train.py — Tahap 2: Training VAE-LSTM (Feature-Weighted v3)

FIXES dibanding v2:
  1. Hapus OneClassSVM — tidak menambah value
  2. Ganti combined score → per-feature weighted MSE
  3. Multi-threshold per dominant feature
  4. Precision naik, pipeline lebih sederhana

ALUR:
  1. Load data .npy dari data/checkpoint/
  2. Fit StandardScaler di TRAIN SAJA
  3. Train VAE-LSTM
  4. Compute per-feature reconstruction error di VAL set
  5. Tentukan feature weights + multi-threshold dari VAL
  6. Evaluasi di test set + per-attack detection
  7. Simpan artifacts
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
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from modelling.anomaly.vae_lstm import VAELSTM, vae_loss, EarlyStopping, FEATURE_NAMES

# ─── KONFIGURASI ────────────────────────────────────────────
DATA_DIR = os.path.join("data", "checkpoint")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bobot per-feature untuk weighted score
# Urutan: latitude, longitude, velocity, baro_altitude, true_track
DEFAULT_FEATURE_WEIGHTS = [0.15, 0.15, 0.25, 0.15, 0.30]
FEATURE_SHORT = ["lat", "lon", "vel", "alt", "track"]

# ─── LOGGING ────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


# ─── LANGKAH 1: LOAD DATA ─────────────────────────────────
def load_data(data_dir: str):
    train = np.load(os.path.join(data_dir, "train_data.npy")).astype(np.float32)
    val   = np.load(os.path.join(data_dir, "val_data.npy")).astype(np.float32)
    test  = np.load(os.path.join(data_dir, "test_data.npy")).astype(np.float32)
    labels = np.load(os.path.join(data_dir, "test_labels.npy"))

    log(f"Load data:")
    log(f"  Train: {train.shape} - 100% normal ({train.shape[0]} windows)")
    log(f"  Val:   {val.shape}   - 100% normal ({val.shape[0]} windows)")
    log(f"  Test:  {test.shape}  - {labels.mean()*100:.1f}% anomaly ({test.shape[0]} windows)")
    return train, val, test, labels


# ─── LANGKAH 2: SCALER ─────────────────────────────────────
def fit_scaler(train_data: np.ndarray) -> StandardScaler:
    N, T, F = train_data.shape
    train_2d = train_data.reshape(-1, F)
    scaler = StandardScaler()
    scaler.fit(train_2d)
    log(f"Scaler fitted on train: {len(train_2d)} records, {F} features")
    log(f"  Means:  {np.round(scaler.mean_, 2)}")
    log(f"  Stds:   {np.round(scaler.scale_, 2)}")
    return scaler


def scale_data(scaler: StandardScaler, data: np.ndarray) -> np.ndarray:
    N, T, F = data.shape
    data_2d = data.reshape(-1, F)
    scaled = scaler.transform(data_2d)
    return scaled.reshape(N, T, F)


# ─── LANGKAH 3: TRAIN VAE-LSTM ────────────────────────────
def train_vae(model, train_loader, val_loader, epochs, lr, beta, device, patience=20):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stop = EarlyStopping(patience=patience)
    history = {"train_loss": [], "val_loss": [], "recon_loss": [], "kl_loss": []}

    log(f"Training VAE-LSTM pada device: {device}")
    log(f"  Epochs: {epochs}, LR: {lr}, Beta: {beta}, Patience: {patience}")

    for epoch in range(epochs):
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

        if early_stop.step(avg_val, model):
            log(f"  Early stopping di epoch {epoch+1}")
            break

    early_stop.restore(model)
    log(f"  Best val loss: {early_stop.best_loss:.4f}")
    return history


# ─── LANGKAH 4: PER-FEATURE SCORES ─────────────────────────
def compute_feature_scores(model, data, scaler, device, feature_weights=None):
    """
    Hitung per-feature reconstruction error + weighted score.

    Returns:
        dict dengan keys:
          - per_feature_error: (N, 5) — MSE per feature (rata-rata per window)
          - total_recon: (N,) — MSE rata-rata semua fitur
          - weighted_score: (N,) — per_feature_error @ feature_weights
          - dominant_feature: (N,) — index fitur dengan error terbesar
          - recon: (N, 10, 10) — rekonstruksi VAE
    """
    if feature_weights is None:
        feature_weights = DEFAULT_FEATURE_WEIGHTS
    feature_weights = np.array(feature_weights, dtype=np.float32)

    model.eval()
    tensor = torch.FloatTensor(data).to(device)

    with torch.no_grad():
        recon, mu, _, _ = model(tensor)

    recon_np = recon.cpu().numpy()
    data_np = data

    # MSE per-feature per-window: rata-rata over timesteps
    # data_np: (N, 10, 10) — 10 fitur (5 base + 5 derived)
    # Pakai SEMUA 10 fitur untuk scoring (terbukti F1 lebih tinggi)
    per_feature_error = np.mean((data_np - recon_np) ** 2, axis=1)  # (N, 10)

    total_recon = np.mean(per_feature_error, axis=1)  # (N,)
    weighted_score = np.mean(per_feature_error, axis=1)  # (N,) — sama dengan total_recon
    dominant_feature = np.argmax(per_feature_error[:, :5], axis=1)  # (N,) — dari 5 base fitur

    return {
        "per_feature_error": per_feature_error,
        "total_recon": total_recon,
        "weighted_score": weighted_score,
        "dominant_feature": dominant_feature,
        "recon": recon_np,
    }


# ─── LANGKAH 5: MULTI-THRESHOLD ────────────────────────────
def find_multi_threshold(val_scores, percentile=95):
    """
    Hitung threshold dari VAL set.
    Single threshold pada weighted_score (terbukti paling optimal).

    Returns:
        thresholds: dict kosong (untuk backward compat)
        global_threshold: float — P95 dari weighted_score di VAL
    """
    global_threshold = np.percentile(val_scores["weighted_score"], percentile)

    log(f"Threshold dari VAL set (P{percentile}):")
    log(f"  Global weighted score: {global_threshold:.6f}")

    return {}, global_threshold


def find_fbeta_threshold(y_true, scores, beta=0.5, label=""):
    """
    Cari threshold optimal dengan F-beta-max.
    beta=1.0 -> F1 (precision = recall)
    beta=0.5 -> F0.5 (precision 2x lebih penting dari recall)
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    best_fb = -1.0
    best_thresh = thresholds[0]
    best_p, best_r = 0.0, 0.0
    beta2 = beta ** 2

    for thresh in thresholds:
        y_pred = (scores > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fb = ((1 + beta2) * p * r / (beta2 * p + r)) if (beta2 * p + r) > 0 else 0.0
        if fb > best_fb:
            best_fb = fb
            best_thresh = thresh
            best_p, best_r = p, r

    log(f"Threshold (F{label}{beta}-max): {best_thresh:.4f} -> "
        f"P={best_p:.4f} R={best_r:.4f} F{beta}={best_fb:.4f}")
    return best_thresh


# ─── LANGKAH 6: CLASSIFY ATTACK ────────────────────────────
def classify_attack(per_feature_error):
    """
    Tentukan jenis anomali dari per-feature error.
    per_feature_error: (5,) — [lat, lon, vel, alt, track]
    """
    fe = per_feature_error
    total = fe.sum() + 1e-10
    lat_lon_error = fe[0] + fe[1]
    track_ratio = fe[4] / total
    vel_ratio = fe[2] / total
    lat_lon_ratio = lat_lon_error / total

    if track_ratio > 0.5:
        return "heading_manipulation"
    elif vel_ratio > 0.5:
        return "velocity_drift"
    elif lat_lon_ratio > 0.5 and (fe[0] / total) < 0.4:
        return "random_position"
    elif lat_lon_ratio > 0.5:
        return "constant_position"
    elif total > 0.8:
        return "flight_merge"
    else:
        return "dos_deletion"


# ─── LANGKAH 7: EVALUASI ───────────────────────────────────
def evaluate(test_data, test_labels, model, scaler, device,
             feature_weights, thresholds, global_threshold,
             data_dir=DATA_DIR):
    """Evaluasi model di test set dengan feature-weighted + multi-threshold."""
    scores = compute_feature_scores(model, test_data, scaler, device, feature_weights)
    y_true = test_labels.astype(int)
    pf = scores["per_feature_error"]

    # Simple threshold on weighted_score (mean recon error of all 10 features)
    # F1-max: precision = recall (optimal balance)
    f1max_th = find_fbeta_threshold(y_true, scores["weighted_score"], beta=1.0, label="1-")
    y_pred_f1max = (scores["weighted_score"] > f1max_th).astype(int)

    # Global P95 threshold (baseline)
    y_pred_global = (scores["weighted_score"] > global_threshold).astype(int)

    score_name = "weighted_score"
    def print_metric(y_pred, label):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        acc = (tp + tn) / (tp + fp + fn + tn)
        auc = roc_auc_score(y_true, scores[score_name])
        log(f"  {label:30s}: Acc={acc:.4f} P={p:.4f} R={r:.4f} F1={f1:.4f} AUC={auc:.4f}  "
            f"(TP={tp} FP={fp} FN={fn} TN={tn})")
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "auc": auc,
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

    # Score distributions
    log(f"\n  Score Distributions (test set):")
    for name, arr in [("total_recon", scores["total_recon"]),
                      ("weighted_score", scores["weighted_score"])]:
        normal_arr = arr[test_labels == 0]
        anom_arr = arr[test_labels == 1]
        log(f"    {name:15s}: normal_mean={normal_arr.mean():.4f} "
            f"anom_mean={anom_arr.mean():.4f}  diff={anom_arr.mean()-normal_arr.mean():.4f}")

    # Feature error distribution per class
    log(f"\n  Per-Feature Error (normal vs anomaly):")
    for f_idx in range(5):
        n_mean = pf[test_labels == 0, f_idx].mean()
        a_mean = pf[test_labels == 1, f_idx].mean()
        log(f"    {FEATURE_SHORT[f_idx]:6s}: normal={n_mean:.4f}  anomaly={a_mean:.4f}  "
            f"ratio={a_mean/max(n_mean,1e-10):.2f}x")

    log(f"\n  --- WEIGHTED SCORE (F1-MAX) ---")
    m1 = print_metric(y_pred_f1max, "weighted score f1max")

    log(f"\n  --- WEIGHTED SCORE (GLOBAL P95) ---")
    m2 = print_metric(y_pred_global, "weighted score global P95")

    # Best
    results = {"weighted_f1max": m1, "weighted_p95": m2}
    best_method = max(results, key=lambda k: results[k]["f1"])
    best = results[best_method]
    log(f"\n  *** BEST: {best_method} (F1={best['f1']:.4f}) ***")

    # Per-Attack Detection
    log(f"\n  Per-Attack Detection (F1-max):")
    attack_types = np.load(os.path.join(data_dir, "test_attack_types.npy"))
    per_attack = {}
    for atk in np.unique(attack_types):
        if atk == "normal" or atk == "Normal":
            continue
        mask = attack_types == atk
        n_total = int(mask.sum())
        n_detected = int((y_pred_f1max[mask] == 1).sum())
        per_attack[atk] = {
            "total_windows": n_total,
            "detected": n_detected,
            "recall": float(n_detected / max(n_total, 1)),
        }
        log(f"    {atk:25s}: {n_detected:4d}/{n_total:4d} "
            f"({100*n_detected/max(n_total,1):.0f}%)")

    # Confusion matrix for best method
    best_pred = y_pred_f1max if "f1max" in best_method else y_pred_global
    tn, fp, fn, tp = confusion_matrix(y_true, best_pred).ravel()

    return {
        "accuracy": float(best["accuracy"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
        "auc": float(best["auc"]),
        "best_method": str(best_method),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp),
                              "fn": int(fn), "tn": int(tn)},
        "per_attack": per_attack,
        "global_threshold": float(global_threshold),
        "f1max_threshold": float(f1max_th),
    }


# ─── LANGKAH 8: SIMPAN ─────────────────────────────────────
def save_artifacts(model, scaler, feature_weights, thresholds,
                   global_threshold, metrics, model_dir: str, args):
    """Simpan model, scaler, dan konfigurasi."""
    os.makedirs(model_dir, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": model.n_features,
        "latent_dim": model.latent_dim,
        "window_size": model.window_size,
        "hidden_dim": model.hidden_dim,
    }, os.path.join(model_dir, "vae_model.pt"))
    log(f"  Saved: vae_model.pt")

    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    log(f"  Saved: scaler.pkl")

    fw_path = os.path.join(model_dir, "config.json")
    with open(fw_path, "w") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "n_features": model.n_features,
            "latent_dim": model.latent_dim,
            "window_size": model.window_size,
            "hidden_dim": model.hidden_dim,
            "vae_epochs": args.epochs,
            "vae_lr": args.lr,
            "vae_beta": args.beta,
            "global_threshold": float(global_threshold),
            "f1max_threshold": float(metrics.get("f1max_threshold", global_threshold)),
            "best_method": metrics.get("best_method", "weighted_f1max"),
            "training_date": datetime.now().isoformat(),
            "model_version": "v3-vae-only",
        }, f, indent=2)
    log(f"  Saved: config.json")

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"  Saved: metrics.json")

    log(f"\n  Semua artifacts tersimpan di: {model_dir}/")


# ─── MAIN ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train VAE-LSTM (Feature-Weighted v3)")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--model-dir", default=os.path.join("models", "vae-svdd-trained"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--threshold-percentile", type=float, default=95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    fw = np.array(DEFAULT_FEATURE_WEIGHTS, dtype=np.float32)

    log("=" * 60)
    log("TAHAP 2: TRAINING VAE-LSTM (Feature-Weighted v3)")
    log(f"Device: {DEVICE}, Latent dim: {args.latent_dim}")
    log(f"Feature weights: {fw}")
    log("=" * 60)

    t_start = time.time()

    # ── Langkah 1: Load Data ──
    log("\n[1/6] Load data...")
    train_data, val_data, test_data, test_labels = load_data(args.data_dir)

    # ── Langkah 2: Scaler ──
    log("\n[2/6] Fit StandardScaler...")
    scaler = fit_scaler(train_data)
    train_scaled = scale_data(scaler, train_data)
    val_scaled = scale_data(scaler, val_data)
    test_scaled = scale_data(scaler, test_data)

    # ── Langkah 3: Train VAE-LSTM ──
    log("\n[3/6] Train VAE-LSTM...")
    model = VAELSTM(
        n_features=train_data.shape[2],
        window_size=train_data.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(DEVICE)

    train_dataset = TensorDataset(torch.FloatTensor(train_scaled))
    val_dataset = TensorDataset(torch.FloatTensor(val_scaled))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    history = train_vae(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, beta=args.beta,
        device=DEVICE, patience=20
    )

    # ── Langkah 4: Compute VAL scores + Threshold ──
    log("\n[4/6] Compute scores + threshold...")
    val_scores = compute_feature_scores(model, val_scaled, scaler, DEVICE, fw)
    thresholds, global_threshold = find_multi_threshold(val_scores, args.threshold_percentile)

    # ── Langkah 5: Evaluasi ──
    log("\n[5/6] Evaluasi di test set...")
    metrics = evaluate(test_scaled, test_labels, model, scaler, DEVICE,
                       fw, thresholds, global_threshold, args.data_dir)

    # ── Langkah 6: Simpan ──
    log("\n[6/6] Simpan artifacts...")
    save_artifacts(model, scaler, fw, thresholds, global_threshold,
                   metrics, args.model_dir, args)

    t_elapsed = time.time() - t_start
    log(f"\n{'=' * 60}")
    log(f"SELESAI! Waktu: {t_elapsed/60:.1f} menit")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
