"""
compare_models_final.py
Bandingkan Model 1 (VAE-LSTM v3) vs Model 2 (Isolation Forest).
"""

import os
import sys
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

VAE_DIR = os.path.join(BASE, "models", "vae-svdd-trained")
IF_DIR = os.path.join(BASE, "models", "isolation-forest")


def load_metrics(model_dir):
    metrics = {}
    config = {}
    metrics_path = os.path.join(model_dir, "metrics.json")
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    return metrics, config


def fmt(val, decimal=4):
    if val is None:
        return "N/A"
    return f"{val:.{decimal}f}"


def pct(val):
    if val is None:
        return "N/A"
    return f"{val*100:.1f}%"


def main():
    vae_metrics, vae_config = load_metrics(VAE_DIR)
    if_metrics, if_config = load_metrics(IF_DIR)

    vae_ok = bool(vae_metrics)
    if_ok = bool(if_metrics)

    print("=" * 60)
    print("  PERBANDINGAN MODEL 1 vs MODEL 2")
    print("=" * 60)

    # ── Config ──
    print(f"\n  Konfigurasi:")
    print(f"  {'Parameter':25s} {'VAE-LSTM v3':>20s} {'IF Standalone':>20s}")
    print(f"  {'-'*25} {'-'*20} {'-'*20}")
    if vae_ok:
        print(f"  {'Model':25s} {'VAE-LSTM':>20s} {'IsolationForest':>20s}")
        print(f"  {'Paradigma':25s} {'Reconstruction-based':>20s} {'Isolation-based':>20s}")
        print(f"  {'Input':25s} {'Sequence (10,10)':>20s} {'Flat (100,)':>20s}")
        print(f"  {'Latent dim':25s} {str(vae_config.get('latent_dim','?')):>20s} {'N/A':>20s}")
        print(f"  {'Threshold':25s} {fmt(vae_metrics.get('f1max_threshold')):>20s} "
              f"{fmt(if_metrics.get('f1max_threshold')):>20s}")

    # ── Metrics ──
    print(f"\n  Metrics Agregat:")
    print(f"  {'Metrik':20s} {'VAE-LSTM v3':>15s} {'IF Standalone':>15s} {'Selisih':>15s}")
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
    fields = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1"),
        ("AUC-ROC", "auc"),
    ]
    for name, key in fields:
        v = vae_metrics.get(key) if vae_ok else None
        i = if_metrics.get(key) if if_ok else None
        if v is not None and i is not None:
            delta = v - i
            delta_str = f"{delta:+.4f}" if isinstance(delta, float) else "N/A"
            print(f"  {name:20s} {fmt(v):>15s} {fmt(i):>15s} {delta_str:>15s}")
        elif v is not None:
            print(f"  {name:20s} {fmt(v):>15s} {'N/A':>15s} {'-':>15s}")
        elif i is not None:
            print(f"  {name:20s} {'N/A':>15s} {fmt(i):>15s} {'-':>15s}")

    # ── Extra info ──
    print(f"\n  Training time:")
    print(f"  {'VAE-LSTM':20s}: ~15 menit")
    print(f"  {'IF Standalone':20s}: ~30 detik")

    # ── Confusion Matrix ──
    print(f"\n  Confusion Matrix (best method):")
    print(f"  {'Metric':10s} {'VAE-LSTM v3':>20s} {'IF Standalone':>20s}")
    print(f"  {'-'*10} {'-'*20} {'-'*20}")
    for key in ["tp", "fp", "fn", "tn"]:
        v = vae_metrics.get("confusion_matrix", {}).get(key) if vae_ok else None
        i = if_metrics.get("confusion_matrix", {}).get(key) if if_ok else None
        print(f"  {key:10s} {str(v) if v else 'N/A':>20s} {str(i) if i else 'N/A':>20s}")

    # ── Per-Attack ──
    print(f"\n  Per-Attack Recall (F1-max):")
    print(f"  {'Attack Type':25s} {'VAE-LSTM':>15s} {'IF':>15s}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")

    vae_atk = vae_metrics.get("per_attack", {}) if vae_ok else {}
    if_atk = if_metrics.get("per_attack", {}) if if_ok else {}

    all_atk = sorted(set(list(vae_atk.keys()) + list(if_atk.keys())))
    for atk in all_atk:
        v_rec = vae_atk.get(atk, {}).get("recall") if vae_ok else None
        i_rec = if_atk.get(atk, {}).get("recall") if if_ok else None
        print(f"  {atk:25s} {pct(v_rec):>15s} {pct(i_rec):>15s}")

    print("=" * 60)
    print(f"  {'VAE-LSTM unggul' if vae_ok and if_ok and vae_metrics.get('f1',0) > if_metrics.get('f1',0) else 'IF unggul'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
