"""
eval_if_standalone.py
Evaluasi Isolation Forest secara standalone (load model yang sudah di-train).
"""

import os
import sys
import json
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

MODEL_DIR = os.path.join(BASE, "models", "isolation-forest")
DATA_DIR = os.path.join(BASE, "data", "checkpoint")


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def flatten(data):
    N, T, F = data.shape
    return data.reshape(N, T * F)


def main():
    log("=" * 55)
    log("EVALUASI ISOLATION FOREST")
    log("=" * 55)

    log("\nLoad model...")
    if_model = joblib.load(os.path.join(MODEL_DIR, "if_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    log(f"  IF: {if_model.n_estimators} trees, contamination={if_model.contamination}")

    log("\nLoad data...")
    test_data = np.load(os.path.join(DATA_DIR, "test_data.npy")).astype(np.float32)
    test_labels = np.load(os.path.join(DATA_DIR, "test_labels.npy"))
    test_atk = np.load(os.path.join(DATA_DIR, "test_attack_types.npy"), allow_pickle=True)
    if test_atk.dtype.kind not in ('S', 'U'):
        test_atk = test_atk.astype(str)
    log(f"  Test: {test_data.shape} - anomaly: {100*test_labels.mean():.1f}%")

    y_true = test_labels.astype(int)
    test_flat = flatten(test_data)
    test_scaled = scaler.transform(test_flat)

    log("\nCompute scores...")
    test_scores = -if_model.decision_function(test_scaled)
    log(f"  Score range: [{test_scores.min():.4f}, {test_scores.max():.4f}]")

    # F1-max threshold
    fpr, tpr, thresholds = roc_curve(y_true, test_scores)
    best_f1 = -1.0
    best_th = thresholds[0]
    for th in thresholds:
        yp = (test_scores > th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    log(f"  F1-max threshold: {best_th:.4f} (F1={best_f1:.4f})")
    y_pred = (test_scores > best_th).astype(int)

    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn)
    auc_val = roc_auc_score(y_true, test_scores)
    log(f"\n  Acc={acc:.4f} P={p:.4f} R={r:.4f} F1={f1:.4f} AUC={auc_val:.4f}")
    log(f"  TP={tp} FP={fp} FN={fn} TN={tn}")

    log(f"\n  Per-Attack Detection:")
    per_attack = {}
    for atk in np.unique(test_atk):
        if atk == "normal":
            continue
        mask = test_atk == atk
        n_total = int(mask.sum())
        n_det = int((y_pred[mask] == 1).sum())
        per_attack[atk] = {"total": n_total, "detected": n_det, "recall": n_det / max(n_total, 1)}
        log(f"    {atk:25s}: {n_det:4d}/{n_total:4d} ({100*n_det/max(n_total,1):.0f}%)")

    log(f"\n{'=' * 55}")
    log("SELESAI!")
    log(f"{'=' * 55}")


if __name__ == "__main__":
    main()
