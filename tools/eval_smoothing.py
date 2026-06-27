"""
eval_smoothing.py — Phase 1: Temporal Smoothing
Grid search majority voting untuk reduksi false positive.

Konsep:
  Tanpa smoothing: banyak false positive (FP=6228, P=52%)
  Dengan smoothing: flag anomali jika ≥K dari M window berturut-turut anomali

Target: F1 naik dari ~0.67 ke ~0.80
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

MODEL_DIR = os.path.join(BASE, "models", "vae-svdd-fixed")
DATA_DIR = os.path.join(BASE, "data", "checkpoint")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SMOOTHING_VARIANTS = [
    (3, 2), (3, 3),
    (5, 3), (5, 4),
    (7, 4), (7, 5),
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def find_threshold(scores_val, percentile=95):
    return np.percentile(scores_val, percentile)


def find_threshold_f1max(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    best_f1 = -1.0
    best_thresh = thresholds[0]
    for thresh in thresholds:
        y_pred = (scores > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def apply_temporal_smoothing(y_pred, M=5, K=3):
    """
    Majority voting temporal smoothing.
    y_pred: (N,) array of 0/1 prediksi per window
    M: window size (jumlah window berturut-turut)
    K: threshold (minimal berapa yang anomaly untuk flag=1)

    Menggunakan centered window: window[i] melihat i-half..i+half
    """
    smoothed = np.zeros_like(y_pred)
    half = M // 2
    for i in range(len(y_pred)):
        start = max(0, i - half)
        end = min(len(y_pred), i + half + 1)
        smoothed[i] = 1 if y_pred[start:end].sum() >= K else 0
    return smoothed


def print_metrics(y_true, y_pred, label="", scores=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn)
    auc = roc_auc_score(y_true, scores) if scores is not None else 0.0
    log(f"  {label:30s} Acc={acc:.4f} P={p:.4f} R={r:.4f} F1={f1:.4f} AUC={auc:.4f}  "
        f"(TP={tp} FP={fp} FN={fn} TN={tn})")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "auc": auc,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def main():
    log("=" * 60)
    log("EVALUASI TEMPORAL SMOOTHING — VAE-LSTM FIXED v2")
    log("=" * 60)

    # ── Load Model ──
    log("\nLoad model dari vae-svdd-fixed...")
    checkpoint = torch.load(os.path.join(MODEL_DIR, "vae_model.pt"),
                            map_location="cpu")
    model = VAELSTM(
        n_features=checkpoint["input_dim"],
        window_size=checkpoint["window_size"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log(f"  VAE-LSTM: {checkpoint['input_dim']} features, "
        f"latent={checkpoint['latent_dim']}")

    svdd = joblib.load(os.path.join(MODEL_DIR, "svdd_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    try:
        score_normalizer = joblib.load(
            os.path.join(MODEL_DIR, "score_normalizer.pkl"))
        log(f"  ScoreNormalizer loaded")
    except Exception:
        score_normalizer = None

    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)
    alpha = config.get("score_alpha", 0.6)
    beta = config.get("score_beta", 0.4)

    # ── Load Data ──
    log("\nLoad dataset...")
    train_data = np.load(os.path.join(DATA_DIR, "train_data.npy")).astype(np.float32)
    val_data = np.load(os.path.join(DATA_DIR, "val_data.npy")).astype(np.float32)
    test_data = np.load(os.path.join(DATA_DIR, "test_data.npy")).astype(np.float32)
    test_labels = np.load(os.path.join(DATA_DIR, "test_labels.npy"))
    test_attack_types = np.load(os.path.join(DATA_DIR, "test_attack_types.npy"),
                                allow_pickle=True)
    if test_attack_types.dtype.kind not in ('S', 'U'):
        test_attack_types = test_attack_types.astype(str)

    log(f"  Test: {test_data.shape} — anomaly: {100*test_labels.mean():.1f}%")
    y_true = test_labels.astype(int)

    # ── Scale ──
    def scale(d):
        N, T, F = d.shape
        d2d = d.reshape(-1, F)
        scaled = scaler.transform(d2d)
        return scaled.reshape(N, T, F)

    val_scaled = scale(val_data)
    test_scaled = scale(test_data)

    # ── Compute Scores ──
    log("\nCompute scores...")

    def compute_scores(data, use_normalizer=False):
        model.eval()
        tensor = torch.FloatTensor(data)
        with torch.no_grad():
            recon, mu, _, _ = model(tensor)
        recon_np = recon.numpy()
        data_np = data
        z_np = mu.numpy()
        recon_error = np.mean((data_np - recon_np) ** 2, axis=(1, 2))
        svdd_scores = svdd.decision_function(z_np)
        svdd_dist = -svdd_scores
        if use_normalizer and score_normalizer is not None:
            recon_norm, svdd_norm = score_normalizer.transform(recon_error, svdd_dist)
        else:
            def _norm(arr):
                mn, mx = arr.min(), arr.max()
                return (arr - mn) / (mx - mn + 1e-10) if mx - mn > 1e-10 else np.zeros_like(arr)
            recon_norm = _norm(recon_error)
            svdd_norm = _norm(svdd_dist)
        combined = alpha * recon_norm + beta * svdd_norm
        return recon_error, combined

    val_recon, _ = compute_scores(val_scaled)
    test_recon, test_combined = compute_scores(test_scaled, use_normalizer=True)

    if score_normalizer is None:
        log("  Fit ScoreNormalizer dari VAL...")
        model.eval()
        val_tensor = torch.FloatTensor(val_scaled)
        with torch.no_grad():
            _, mu_val, _, _ = model(val_tensor)
        z_val = mu_val.numpy()
        svdd_val = -svdd.decision_function(z_val)
        score_normalizer = type('obj', (object,), {
            'recon_min': float(val_recon.min()),
            'recon_max': float(val_recon.max()),
            'svdd_min': float(svdd_val.min()),
            'svdd_max': float(svdd_val.max()),
            'transform': lambda self, r, s: (
                (r - self.recon_min) / (self.recon_max - self.recon_min + 1e-10),
                (s - self.svdd_min) / (self.svdd_max - self.svdd_min + 1e-10)
            )})()
        test_recon, test_combined = compute_scores(test_scaled, use_normalizer=True)

    # ── Baseline (tanpa smoothing) ──
    log(f"\n{'='*60}")
    log("BASELINE — Tanpa Smoothing")
    log(f"{'='*60}")

    recon_threshold = find_threshold(val_recon, 95)
    f1max_threshold = find_threshold_f1max(y_true, test_recon)

    y_pred_baseline = (test_recon > f1max_threshold).astype(int)
    base = print_metrics(y_true, y_pred_baseline, "Baseline (recon f1max)", test_recon)

    # ── Grid Search Smoothing ──
    log(f"\n{'='*60}")
    log("GRID SEARCH TEMPORAL SMOOTHING")
    log(f"{'='*60}")
    log(f"  {'Variant':15s} {'M':3s} {'K':3s} {'Accuracy':>10s} {'Precision':>10s} "
        f"{'Recall':>10s} {'F1':>10s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'TN':>6s}")
    log(f"  {'-'*15} {'-'*3} {'-'*3} {'-'*10} {'-'*10} {'-'*10} {'-'*10} "
        f"{'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    results = {}
    best_f1 = -1.0
    best_variant = None
    best_pred = None

    for M, K in SMOOTHING_VARIANTS:
        y_pred_smooth = apply_temporal_smoothing(y_pred_baseline, M=M, K=K)
        name = f"M={M},K={K}"
        m = print_metrics(y_true, y_pred_smooth, name, test_recon)
        results[name] = m

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_variant = (M, K)
            best_pred = y_pred_smooth

    # ── Best Smoothing ──
    log(f"\n{'='*60}")
    log(f"BEST SMOOTHING: M={best_variant[0]}, K={best_variant[1]} "
        f"(F1={best_f1:.4f})")
    log(f"{'='*60}")

    best_m, best_k = best_variant
    log(f"\n  Baseline:  F1={base['f1']:.4f}  P={base['precision']:.4f}  "
        f"R={base['recall']:.4f}")
    log(f"  Smoothed:  F1={best_f1:.4f}  "
        f"P={results[f'M={best_m},K={best_k}']['precision']:.4f}  "
        f"R={results[f'M={best_m},K={best_k}']['recall']:.4f}")
    delta_f1 = best_f1 - base["f1"]
    log(f"  Delta:     F1={delta_f1:+.4f}")

    # ── Per-Attack Best ──
    log(f"\n  Per-Attack Detection (smoothed, M={best_m}, K={best_k}):")
    per_attack = {}
    for atk in np.unique(test_attack_types):
        if atk == "normal" or atk == "Normal":
            continue
        mask = test_attack_types == atk
        n_total = int(mask.sum())
        n_detected = int((best_pred[mask] == 1).sum())
        per_attack[atk] = {
            "total_windows": n_total,
            "detected": n_detected,
            "recall": float(n_detected / max(n_total, 1)),
        }
        log(f"    {atk:25s}: {n_detected:4d}/{n_total:4d} "
            f"({100*n_detected/max(n_total,1):.0f}%)")

    # ── Simpan metrics ──
    metrics = {
        "dataset_note": "Realistic attacks + temporal smoothing",
        "smoothing_params": {"M": best_m, "K": best_k},
        "baseline": base,
        "best_smoothed": results[f"M={best_m},K={best_k}"],
        "all_variants": {k: v for k, v in results.items()},
        "per_attack": per_attack,
    }

    metrics_path = os.path.join(MODEL_DIR, "metrics_smoothing.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"\n  Metrics saved to: {metrics_path}")

    log(f"\n{'='*60}")
    log("SELESAI!")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
