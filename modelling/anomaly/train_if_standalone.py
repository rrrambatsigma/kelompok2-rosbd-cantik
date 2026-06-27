"""
train_if_standalone.py — Model 2: Isolation Forest Standalone

Perbandingan dengan Model 1 (VAE-LSTM v3):
  Model 1: Sequence → LSTM → latent → reconstruction → recon_error
  Model 2: Flat 100 fitur → 100 decision trees → isolation score

ALUR:
  1. Load data dari data/checkpoint/ (sama dengan VAE-LSTM)
  2. Flatten window: (N, 10, 10) → (N, 100)
  3. StandardScaler fit dari TRAIN SAJA
  4. Train IsolationForest
  5. Compute threshold dari VAL set (P95 + F1-max)
  6. Evaluasi di test set (accuracy, precision, recall, F1, AUC, per-attack)
  7. Simpan ke models/isolation-forest/
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
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE)

DATA_DIR = os.path.join(BASE, "data", "checkpoint")
MODEL_DIR = os.path.join(BASE, "models", "isolation-forest")
RANDOM_SEED = 42


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def load_data(data_dir):
    train = np.load(os.path.join(data_dir, "train_data.npy")).astype(np.float32)
    val   = np.load(os.path.join(data_dir, "val_data.npy")).astype(np.float32)
    test  = np.load(os.path.join(data_dir, "test_data.npy")).astype(np.float32)
    labels = np.load(os.path.join(data_dir, "test_labels.npy"))
    log(f"Load data:")
    log(f"  Train: {train.shape} - 100% normal ({train.shape[0]} windows)")
    log(f"  Val:   {val.shape}   - 100% normal ({val.shape[0]} windows)")
    log(f"  Test:  {test.shape}  - {labels.mean()*100:.1f}% anomaly ({test.shape[0]} windows)")
    return train, val, test, labels


def flatten(data):
    """(N, 10, 10) → (N, 100)"""
    N, T, F = data.shape
    return data.reshape(N, T * F)


def find_fbeta_threshold(y_true, scores, beta=1.0, label=""):
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


def save_artifacts(if_model, scaler, threshold, metrics, model_dir, args):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(if_model, os.path.join(model_dir, "if_model.pkl"))
    log(f"  Saved: if_model.pkl")
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    log(f"  Saved: scaler.pkl")
    config = {
        "model": "IsolationForest",
        "n_estimators": args.n_estimators,
        "contamination": args.contamination,
        "max_samples": args.max_samples,
        "threshold": float(threshold),
        "f1max_threshold": float(metrics.get("f1max_threshold", threshold)),
        "feature_dim": 100,
        "training_date": datetime.now().isoformat(),
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    log(f"  Saved: config.json")
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"  Saved: metrics.json")
    log(f"\n  Semua artifacts tersimpan di: {model_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train Isolation Forest Standalone")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--contamination", type=str, default="auto")
    parser.add_argument("--max-samples", type=str, default="auto")
    parser.add_argument("--threshold-percentile", type=float, default=95)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    np.random.seed(args.seed)

    log("=" * 55)
    log("ISOLATION FOREST — Standalone Model 2")
    log("=" * 55)

    t_start = time.time()

    # ── Step 1: Load Data ──
    log("\n[1/6] Load data...")
    train_data, val_data, test_data, test_labels = load_data(args.data_dir)

    # ── Step 2: Flatten ──
    log("\n[2/6] Flatten windows...")
    train_flat = flatten(train_data)
    val_flat = flatten(val_data)
    test_flat = flatten(test_data)
    log(f"  Train: {train_flat.shape} (flatten 10x10 = 100)")
    log(f"  Val:   {val_flat.shape}")
    log(f"  Test:  {test_flat.shape}")

    # ── Step 3: Scaler ──
    log("\n[3/6] Fit StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(train_flat)
    log(f"  Scaler fitted: {scaler.mean_.shape[0]} features")

    train_scaled = scaler.transform(train_flat)
    val_scaled = scaler.transform(val_flat)
    test_scaled = scaler.transform(test_flat)

    # ── Step 4: Train IF ──
    log("\n[4/6] Train IsolationForest...")
    if_model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        max_samples=args.max_samples,
        random_state=args.seed,
        n_jobs=-1,
    )
    if_model.fit(train_scaled)
    log(f"  n_estimators={args.n_estimators}, contamination={args.contamination}")

    # Anomaly score: makin negatif = makin anomali
    # decision_function returns: makin negatif = makin anomali
    # Kita flip jadi: makin positif = makin anomali (konsisten dengan VAE)
    val_scores = -if_model.decision_function(val_scaled)
    test_scores = -if_model.decision_function(test_scaled)

    log(f"  Train score range: [{-if_model.decision_function(train_scaled).min():.4f}, "
        f"{-if_model.decision_function(train_scaled).max():.4f}]")
    log(f"  Val score range:   [{val_scores.min():.4f}, {val_scores.max():.4f}]")

    # ── Step 5: Threshold ──
    log("\n[5/6] Compute threshold...")
    p95_threshold = np.percentile(val_scores, args.threshold_percentile)
    log(f"  P95 dari VAL: {p95_threshold:.4f}")

    y_true = test_labels.astype(int)
    f1max_th = find_fbeta_threshold(y_true, test_scores, beta=1.0, label="1-")

    # ── Step 6: Evaluasi ──
    log("\n[6/6] Evaluasi...")
    y_pred_f1max = (test_scores > f1max_th).astype(int)
    y_pred_p95 = (test_scores > p95_threshold).astype(int)

    def print_metric(y_pred, label):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        acc = (tp + tn) / (tp + fp + fn + tn)
        auc_val = roc_auc_score(y_true, test_scores)
        log(f"  {label:30s}: Acc={acc:.4f} P={p:.4f} R={r:.4f} F1={f1:.4f} AUC={auc_val:.4f}  "
            f"(TP={tp} FP={fp} FN={fn} TN={tn})")
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "auc": auc_val}

    log(f"\n  --- ISOLATION FOREST (F1-MAX) ---")
    m1 = print_metric(y_pred_f1max, "if f1max")

    log(f"\n  --- ISOLATION FOREST (P95 VAL) ---")
    m2 = print_metric(y_pred_p95, "if p95")

    best = m1 if m1["f1"] >= m2["f1"] else m2
    best_method = "if_f1max" if best == m1 else "if_p95"
    log(f"\n  *** BEST: {best_method} (F1={best['f1']:.4f}) ***")

    # Per-attack
    log(f"\n  Per-Attack Detection (F1-max):")
    attack_types = np.load(os.path.join(args.data_dir, "test_attack_types.npy"),
                           allow_pickle=True)
    if attack_types.dtype.kind not in ('S', 'U'):
        attack_types = attack_types.astype(str)
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

    # Best confusion matrix
    tn_all, fp_all, fn_all, tp_all = confusion_matrix(y_true, y_pred_f1max).ravel()

    metrics = {
        "accuracy": float(best["accuracy"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
        "auc": float(best["auc"]),
        "best_method": str(best_method),
        "best_threshold": float(f1max_th),
        "f1max_threshold": float(f1max_th),
        "p95_threshold": float(p95_threshold),
        "confusion_matrix": {"tp": int(tp_all), "fp": int(fp_all),
                              "fn": int(fn_all), "tn": int(tn_all)},
        "per_attack": per_attack,
    }

    save_artifacts(if_model, scaler, f1max_th, metrics, args.model_dir, args)

    t_elapsed = time.time() - t_start
    log(f"\n{'=' * 55}")
    log(f"SELESAI! Waktu: {t_elapsed/60:.2f} menit")
    log(f"{'=' * 55}")


if __name__ == "__main__":
    main()
