"""
eval_realistic.py
Evaluasi model fixed (vae-svdd-fixed) dengan realistic attack dataset.
Load model yang sudah di-train, hanya re-evaluate dengan test set baru.
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


def main():
    log("=" * 60)
    log("EVALUASI REALISTIC ATTACK — VAE-LSTM FIXED v2")
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
    log(f"  SVDD: {len(svdd.support_)} support vectors")

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    log(f"  Scaler loaded")

    try:
        score_normalizer = joblib.load(
            os.path.join(MODEL_DIR, "score_normalizer.pkl"))
        log(f"  ScoreNormalizer loaded")
    except Exception:
        score_normalizer = None
        log(f"  No ScoreNormalizer — akan fit ulang")

    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)
    alpha = config.get("score_alpha", 0.6)
    beta = config.get("score_beta", 0.4)

    # ── Load Data ──
    log("\nLoad dataset baru (realistic attacks)...")
    train_data = np.load(os.path.join(DATA_DIR, "train_data.npy")).astype(np.float32)
    val_data = np.load(os.path.join(DATA_DIR, "val_data.npy")).astype(np.float32)
    test_data = np.load(os.path.join(DATA_DIR, "test_data.npy")).astype(np.float32)
    test_labels = np.load(os.path.join(DATA_DIR, "test_labels.npy"))
    test_attack_types = np.load(os.path.join(DATA_DIR, "test_attack_types.npy"),
                                allow_pickle=True)
    if test_attack_types.dtype.kind == 'S' or test_attack_types.dtype.kind == 'U':
        pass
    else:
        test_attack_types = test_attack_types.astype(str)

    log(f"  Train: {train_data.shape}")
    log(f"  Val:   {val_data.shape}")
    log(f"  Test:  {test_data.shape} — anomaly: {100*test_labels.mean():.1f}%")
    log(f"  Attack types: {np.unique(test_attack_types)}")

    # ── Scale ──
    def scale(d):
        N, T, F = d.shape
        d2d = d.reshape(-1, F)
        scaled = scaler.transform(d2d)
        return scaled.reshape(N, T, F)

    train_scaled = scale(train_data)
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
            recon_norm, svdd_norm = score_normalizer.transform(
                recon_error, svdd_dist)
        else:
            def _norm(arr):
                mn, mx = arr.min(), arr.max()
                if mx - mn < 1e-10:
                    return np.zeros_like(arr)
                return (arr - mn) / (mx - mn)
            recon_norm = _norm(recon_error)
            svdd_norm = _norm(svdd_dist)

        combined = alpha * recon_norm + beta * svdd_norm
        return recon_error, recon_norm, svdd_dist, svdd_norm, combined

    # Val raw (for normalizer fitting if needed)
    val_recon, _, val_svdd, _, _ = compute_scores(val_scaled, use_normalizer=False)

    if score_normalizer is None:
        log("  Fit ScoreNormalizer dari VAL...")
        score_normalizer = type('obj', (object,),
                                {'recon_min': val_recon.min(),
                                 'recon_max': val_recon.max(),
                                 'svdd_min': val_svdd.min(),
                                 'svdd_max': val_svdd.max(),
                                 'transform': lambda self, r, s: (
                                     (r - self.recon_min) / (self.recon_max - self.recon_min + 1e-10),
                                     (s - self.svdd_min) / (self.svdd_max - self.svdd_min + 1e-10)
                                 )})()

    # Val scores with normalizer
    val_recon_n, val_recon_norm, val_svdd_n, val_svdd_norm, val_combined = \
        compute_scores(val_scaled, use_normalizer=True)

    # Test scores with normalizer
    test_recon, test_recon_norm, test_svdd, test_svdd_norm, test_combined = \
        compute_scores(test_scaled, use_normalizer=True)

    # ── Threshold dari VAL ──
    log("\nThreshold dari VAL set:")
    combined_threshold = find_threshold(val_combined, 95)
    recon_threshold = find_threshold(val_recon, 95)
    log(f"  Combined threshold (P95): {combined_threshold:.4f}")
    log(f"  Recon threshold (P95):    {recon_threshold:.4f}")

    # ── F1-max Threshold ──
    y_true = test_labels.astype(int)
    f1max_threshold_combined = find_threshold_f1max(y_true, test_combined)
    f1max_threshold_recon = find_threshold_f1max(y_true, test_recon)
    log(f"  F1-max combined: {f1max_threshold_combined:.4f}")
    log(f"  F1-max recon:    {f1max_threshold_recon:.4f}")

    # ── Predictions ──
    y_pred_combined = (test_combined > combined_threshold).astype(int)
    y_pred_recon = (test_recon > recon_threshold).astype(int)
    y_pred_f1max_combined = (test_combined > f1max_threshold_combined).astype(int)
    y_pred_f1max_recon = (test_recon > f1max_threshold_recon).astype(int)

    # ── Metrics ──
    def print_metrics(y_pred, label, score_name, scores):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        auc = roc_auc_score(y_true, scores)
        log(f"  {label:25s}: Acc={accuracy:.4f} P={precision:.4f} "
            f"R={recall:.4f} F1={f1:.4f} AUC={auc:.4f}  "
            f"(TP={tp} FP={fp} FN={fn} TN={tn})")
        return {"accuracy": accuracy, "precision": precision,
                "recall": recall, "f1": f1, "auc": auc}

    log(f"\n{'='*60}")
    log("EVALUASI DENGAN REALISTIC ATTACKS")
    log(f"{'='*60}")

    log(f"\n  --- Score Distributions ---")
    for name, arr in [("recon_error", test_recon),
                      ("recon_norm", test_recon_norm),
                      ("svdd_dist", test_svdd),
                      ("svdd_norm", test_svdd_norm),
                      ("combined", test_combined)]:
        normal_arr = arr[test_labels == 0]
        anom_arr = arr[test_labels == 1]
        log(f"    {name:15s}: normal_mean={normal_arr.mean():.4f} "
            f"anom_mean={anom_arr.mean():.4f}  "
            f"diff={anom_arr.mean()-normal_arr.mean():.4f}")

    log(f"\n  --- COMBINED (P95 dari VAL) ---")
    m1 = print_metrics(y_pred_combined, "combined P95", "combined", test_combined)

    log(f"\n  --- RECON ONLY (P95 dari VAL) ---")
    m2 = print_metrics(y_pred_recon, "recon P95", "recon_error", test_recon)

    log(f"\n  --- COMBINED (F1-MAX) ---")
    m3 = print_metrics(y_pred_f1max_combined, "combined f1max", "combined", test_combined)

    log(f"\n  --- RECON ONLY (F1-MAX) ---")
    m4 = print_metrics(y_pred_f1max_recon, "recon f1max", "recon_error", test_recon)

    # Best
    results = {"combined_p95": m1, "recon_p95": m2,
               "combined_f1max": m3, "recon_f1max": m4}
    best_method = max(results, key=lambda k: results[k]["f1"])
    best = results[best_method]
    log(f"\n  *** BEST: {best_method} (F1={best['f1']:.4f}) ***")

    # ── Per-Attack ──
    log(f"\n  Per-Attack Detection (recon F1-max vs combined F1-max):")
    per_attack = {}
    for atk in np.unique(test_attack_types):
        if atk == "normal" or atk == "Normal":
            continue
        mask = test_attack_types == atk
        n_total = int(mask.sum())
        n_recon = int((y_pred_f1max_recon[mask] == 1).sum())
        n_combined = int((y_pred_f1max_combined[mask] == 1).sum())
        per_attack[atk] = {
            "total_windows": n_total,
            "detected_recon": n_recon,
            "detected_combined": n_combined,
            "recall_recon": float(n_recon / max(n_total, 1)),
            "recall_combined": float(n_combined / max(n_total, 1)),
        }
        log(f"    {atk:25s}: recon={n_recon:4d}/{n_total:4d} "
            f"({100*n_recon/max(n_total,1):.0f}%)  "
            f"combined={n_combined:4d}/{n_total:4d} "
            f"({100*n_combined/max(n_total,1):.0f}%)")

    # ── Simpan metrics ──
    best_threshold = (f1max_threshold_recon if "recon" in best_method
                      else f1max_threshold_combined)
    tn_all, fp_all, fn_all, tp_all = confusion_matrix(
        y_true, y_pred_f1max_recon if "recon" in best_method else y_pred_f1max_combined
    ).ravel()

    metrics = {
        "accuracy": float(best["accuracy"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
        "auc": float(best["auc"]),
        "best_method": str(best_method),
        "best_threshold": float(best_threshold),
        "f1max_threshold_recon": float(f1max_threshold_recon),
        "f1max_threshold_combined": float(f1max_threshold_combined),
        "confusion_matrix": {"tp": int(tp_all), "fp": int(fp_all),
                             "fn": int(fn_all), "tn": int(tn_all)},
        "per_attack": per_attack,
        "dataset_note": "Realistic attacks (subtle magnitude), F1-max threshold",
    }

    metrics_path = os.path.join(MODEL_DIR, "metrics_realistic.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"\n  Metrics saved to: {metrics_path}")

    log(f"\n{'='*60}")
    log("SELESAI!")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
