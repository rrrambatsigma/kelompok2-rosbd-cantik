"""
compare_vae_versions.py
Bandingkan performa VAE-LSTM v1 (old) vs v2 (fixed).

Membaca metrics.json dari:
  - models/vae-svdd/       (v1, old)
  - models/vae-svdd-fixed/ (v2, fixed)
"""

import os
import sys
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

OLD_DIR = os.path.join(BASE, "models", "vae-svdd")
FIXED_DIR = os.path.join(BASE, "models", "vae-svdd-fixed")


def load_metrics(model_dir):
    metrics_path = os.path.join(model_dir, "metrics.json")
    config_path = os.path.join(model_dir, "config.json")
    metrics = {}
    config = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    return metrics, config


def print_separator(char="=", n=60):
    print(char * n)


def main():
    old_metrics, old_config = load_metrics(OLD_DIR)
    fixed_metrics, fixed_config = load_metrics(FIXED_DIR)

    if not old_metrics and not fixed_metrics:
        print("Tidak ada metrics ditemukan. Jalankan training dulu.")
        return

    print_separator()
    print("  PERBANDINGAN VAE-LSTM: OLD (v1) vs FIXED (v2)")
    print_separator()
    print()

    old_available = bool(old_metrics)
    fixed_available = bool(fixed_metrics)

    # ── Konfigurasi ──
    print("  Konfigurasi:")
    print(f"  {'Parameter':25s} {'OLD':>25s} {'FIXED':>25s}")
    print(f"  {'-'*25} {'-'*25} {'-'*25}")
    params = ["latent_dim", "svdd_nu", "svdd_gamma",
              "score_alpha", "score_beta", "svdd_grid_search"]
    for p in params:
        ov = old_config.get(p, "N/A") if old_available else "N/A"
        fv = fixed_config.get(p, "N/A") if fixed_available else "N/A"
        print(f"  {p:25s} {str(ov):>25s} {str(fv):>25s}")
    print()

    # ── Metrics agregat ──
    print("  Metrics Agregat:")
    print(f"  {'Metrik':20s} {'OLD':>12s} {'FIXED':>12s} {'Delta':>12s}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    metrics_fields = ["accuracy", "precision", "recall", "f1", "auc"]
    for m in metrics_fields:
        ov = old_metrics.get(m, None) if old_available else None
        fv = fixed_metrics.get(m, None) if fixed_available else None
        if ov is not None and fv is not None:
            delta = fv - ov
            print(f"  {m:20s} {ov:>12.4f} {fv:>12.4f} {delta:+>12.4f}")
        elif ov is not None:
            print(f"  {m:20s} {ov:>12.4f} {'N/A':>12s} {'N/A':>12s}")
        elif fv is not None:
            print(f"  {m:20s} {'N/A':>12s} {fv:>12.4f} {'N/A':>12s}")
        else:
            print(f"  {m:20s} {'N/A':>12s} {'N/A':>12s} {'N/A':>12s}")

    # ── Best method ──
    print()
    if old_available:
        print(f"  OLD   best_method: {old_metrics.get('best_method', 'N/A')}  "
              f"threshold={old_metrics.get('best_threshold', 'N/A')}")
    if fixed_available:
        print(f"  FIXED best_method: {fixed_metrics.get('best_method', 'N/A')}  "
              f"threshold={fixed_metrics.get('best_threshold', 'N/A')}")

    # ── Youden threshold ──
    print()
    if old_available and fixed_available:
        orc = old_metrics.get('youden_threshold_recon', 'N/A')
        occ = old_metrics.get('youden_threshold_combined', 'N/A')
        frc = fixed_metrics.get('youden_threshold_recon', 'N/A')
        fcc = fixed_metrics.get('youden_threshold_combined', 'N/A')
        print(f"  Youden threshold recon:    OLD={orc}  FIXED={frc}")
        print(f"  Youden threshold combined: OLD={occ}  FIXED={fcc}")

    # ── Per-Attack ──
    print()
    print("  Per-Attack Detection Recall (combined Youden):")
    print(f"  {'Attack Type':25s} {'OLD':>12s} {'FIXED':>12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")

    old_attacks = old_metrics.get("per_attack", {}) if old_available else {}
    fixed_attacks = fixed_metrics.get("per_attack", {}) if fixed_available else {}

    all_attacks = sorted(set(list(old_attacks.keys()) + list(fixed_attacks.keys())))
    for atk in all_attacks:
        o_rec = old_attacks.get(atk, {}).get("recall_combined", None) if old_available else None
        f_rec = fixed_attacks.get(atk, {}).get("recall_combined", None) if fixed_available else None

        o_str = f"{o_rec*100:.1f}%" if o_rec is not None else "N/A"
        f_str = f"{f_rec*100:.1f}%" if f_rec is not None else "N/A"
        o_det = old_attacks.get(atk, {}).get("detected_combined", None) if old_available else None
        f_det = fixed_attacks.get(atk, {}).get("detected_combined", None) if fixed_available else None
        o_tot = old_attacks.get(atk, {}).get("total_windows", None) if old_available else None
        f_tot = fixed_attacks.get(atk, {}).get("total_windows", None) if fixed_available else None

        if o_det is not None and o_tot is not None:
            o_str = f"{o_det}/{o_tot} ({o_rec*100:.1f}%)"
        if f_det is not None and f_tot is not None:
            f_str = f"{f_det}/{f_tot} ({f_rec*100:.1f}%)"
        print(f"  {atk:25s} {o_str:>12s} {f_str:>12s}")

    # ── Confusion Matrix ──
    print()
    print("  Confusion Matrix (best method):")
    print(f"  {'Metric':15s} {'OLD':>12s} {'FIXED':>12s}")
    print(f"  {'-'*15} {'-'*12} {'-'*12}")
    for cm_key in ["tp", "fp", "fn", "tn"]:
        ov = old_metrics.get("confusion_matrix", {}).get(cm_key, None) if old_available else None
        fv = fixed_metrics.get("confusion_matrix", {}).get(cm_key, None) if fixed_available else None
        print(f"  {cm_key:15s} {str(ov) if ov is not None else 'N/A':>12s} "
              f"{str(fv) if fv is not None else 'N/A':>12s}")

    print_separator()


if __name__ == "__main__":
    main()
