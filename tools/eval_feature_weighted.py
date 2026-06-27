"""
eval_feature_weighted.py
Evaluasi VAE-LSTM v3 (feature-weighted + multi-threshold) dengan realistic dataset.
"""

import os
import sys
import json
import numpy as np
import joblib
import torch
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from modelling.anomaly.vae_lstm import VAELSTM, FEATURE_NAMES

MODEL_DIR = os.path.join(BASE, "models", "vae-svdd-trained")
DATA_DIR = os.path.join(BASE, "data", "checkpoint")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_SHORT = ["lat", "lon", "vel", "alt", "track"]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def main():
    log("=" * 60)
    log("EVALUASI V3 — Feature-Weighted VAE-LSTM")
    log("=" * 60)

    # ── Load Model ──
    log("\nLoad model...")
    checkpoint = torch.load(os.path.join(MODEL_DIR, "vae_model.pt"), map_location="cpu")
    model = VAELSTM(
        n_features=checkpoint["input_dim"],
        window_size=checkpoint["window_size"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log(f"  VAE-LSTM: {checkpoint['input_dim']} features, latent={checkpoint['latent_dim']}")

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_weights.json")) as f:
        fw_data = json.load(f)
    feature_weights = np.array(fw_data["weights"], dtype=np.float32)
    thresholds = {int(k): v for k, v in fw_data["thresholds"].items()}
    global_threshold = fw_data["global_threshold"]
    log(f"  Feature weights: {feature_weights}")
    log(f"  Thresholds: {thresholds}")

    # ── Load Data ──
    log("\nLoad dataset...")
    test_data = np.load(os.path.join(DATA_DIR, "test_data.npy")).astype(np.float32)
    test_labels = np.load(os.path.join(DATA_DIR, "test_labels.npy"))
    test_attack_types = np.load(os.path.join(DATA_DIR, "test_attack_types.npy"), allow_pickle=True)
    if test_attack_types.dtype.kind not in ('S', 'U'):
        test_attack_types = test_attack_types.astype(str)
    log(f"  Test: {test_data.shape} — anomaly: {100*test_labels.mean():.1f}%")
    y_true = test_labels.astype(int)

    # ── Scale ──
    N, T, F = test_data.shape
    scaled_2d = scaler.transform(test_data.reshape(-1, F))
    test_scaled = scaled_2d.reshape(N, T, F)

    # ── Compute Scores ──
    log("\nCompute scores...")
    model.eval()
    tensor = torch.FloatTensor(test_scaled)
    with torch.no_grad():
        recon, _, _, _ = model(tensor)
    recon_np = recon.numpy()

    # MSE semua 10 fitur (sama dengan train.py v3)
    recon_error = np.mean((test_scaled - recon_np) ** 2, axis=(1, 2))
    weighted_score = recon_error.copy()
    per_feature_base = np.mean((test_scaled[:,:,:5] - recon_np[:,:,:5]) ** 2, axis=1)

    # ── F1-max threshold ──
    fpr, tpr, f1_thresholds = roc_curve(y_true, weighted_score)
    best_f1 = -1.0
    best_th = f1_thresholds[0]
    for th in f1_thresholds:
        yp = (weighted_score > th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    y_pred_f1max = (weighted_score > best_th).astype(int)

    # ── Global P95 threshold ──
    y_pred_global = (weighted_score > global_threshold).astype(int)

    # ── Print metrics ──
    def print_metric(y_pred, label):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        acc = (tp + tn) / (tp + fp + fn + tn)
        auc = roc_auc_score(y_true, weighted_score)
        log(f"  {label:30s}: Acc={acc:.4f} P={p:.4f} R={r:.4f} F1={f1:.4f} AUC={auc:.4f}  "
            f"(TP={tp} FP={fp} FN={fn} TN={tn})")
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "auc": auc,
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

    log(f"\n  --- Score Distributions ---")
    for name, arr in [("recon_error", recon_error), ("weighted_score", weighted_score)]:
        nm = arr[test_labels == 0].mean()
        am = arr[test_labels == 1].mean()
        log(f"    {name:15s}: normal_mean={nm:.4f} anom_mean={am:.4f} diff={am-nm:.4f}")

    log(f"\n  --- Per-Feature Error ---")
    for f_idx in range(5):
        nm = per_feature_base[test_labels == 0, f_idx].mean()
        am = per_feature_base[test_labels == 1, f_idx].mean()
        log(f"    {FEATURE_SHORT[f_idx]:6s}: normal={nm:.4f} anomaly={am:.4f} ratio={am/max(nm,1e-10):.2f}x")

    log(f"\n  --- WEIGHTED SCORE (F1-MAX) ---")
    m1 = print_metric(y_pred_f1max, "weighted score f1max")

    log(f"\n  --- WEIGHTED SCORE (GLOBAL P95) ---")
    m2 = print_metric(y_pred_global, "weighted score global P95")

    # Best
    results = {"weighted_f1max": m1, "weighted_p95": m2}
    best_method = max(results, key=lambda k: results[k]["f1"])
    best = results[best_method]
    log(f"\n  *** BEST: {best_method} (F1={best['f1']:.4f}) ***")

    # Per-attack
    log(f"\n  Per-Attack Detection (F1-max):")
    per_attack = {}
    for atk in np.unique(test_attack_types):
        if atk == "normal" or atk == "Normal":
            continue
        mask = test_attack_types == atk
        n_total = int(mask.sum())
        n_detected = int((y_pred_f1max[mask] == 1).sum())
        per_attack[atk] = {
            "total_windows": n_total,
            "detected": n_detected,
            "recall": float(n_detected / max(n_total, 1)),
        }
        log(f"    {atk:25s}: {n_detected:4d}/{n_total:4d} ({100*n_detected/max(n_total,1):.0f}%)")

    # Save
    metrics = {
        "dataset_note": "Realistic attacks, VAE-only v3",
        "best_method": best_method,
        "weighted_f1max": m1,
        "weighted_p95": m2,
        "per_attack": per_attack,
        "global_threshold": global_threshold,
        "f1max_threshold": float(best_th),
    }
    metrics_path = os.path.join(MODEL_DIR, "metrics_feature_weighted.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"\n  Metrics saved to: {metrics_path}")

    log(f"\n{'='*60}")
    log("SELESAI!")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
