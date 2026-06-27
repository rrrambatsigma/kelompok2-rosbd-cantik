"""
reinject_attacks.py
Mengganti attack di test set dengan realistic magnitude.
Data di-load dari checkpoint yang sudah ada (tanpa ES).

Alur:
  1. Load val_data.npy (100% normal windows) sebagai basis test set
  2. Inject realistic attacks langsung di level window (10 timesteps)
  3. Recompute derivative features setelah modifikasi
  4. Simpan sebagai test_data.npy, test_labels.npy, test_attack_types.npy
"""

import os
import sys
import numpy as np
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

DATA_DIR = os.path.join(BASE, "data", "checkpoint")
OUTPUT_DIR = os.path.join(BASE, "data", "checkpoint")
RANDOM_SEED = 42

FEATURES = ["latitude", "longitude", "velocity", "baro_altitude", "true_track"]
DERIVED = ["dlat", "dlon", "dvel", "dalt", "dtrack"]
WINDOW_SIZE = 10

ATTACK_FRACTIONS = {
    "constant_position": 0.08,
    "random_position": 0.08,
    "velocity_drift": 0.06,
    "dos_deletion": 0.05,
    "flight_merge": 0.05,
    "heading_manipulation": 0.06,
}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def recompute_derived(base_window):
    """
    base_window: (10, 5) — [lat, lon, vel, alt, track]
    return: (10, 10) — [lat, lon, vel, alt, track, dlat, dlon, dvel, dalt, dtrack]
    """
    deltas = np.diff(base_window, axis=0, prepend=base_window[0:1])
    return np.concatenate([base_window, deltas], axis=1)


def inject_attack(window_10f, attack_type, rng, donor_window=None):
    """
    Inject attack ke window (10, 10).
    window_10f: (10, 10) — sudah termasuk derived features
    return: (10, 10) yang sudah dimodifikasi + label 0/1
    """
    # Ekstrak 5 base features
    base = window_10f[:, :5].copy()

    if attack_type == "constant_position":
        # GPS spoofing: posisi stuck + noise kecil
        base[:, 0] = base[0, 0] + rng.normal(0, 0.005, size=WINDOW_SIZE)  # lat
        base[:, 1] = base[0, 1] + rng.normal(0, 0.005, size=WINDOW_SIZE)  # lon

    elif attack_type == "random_position":
        # GPS jitter: noise kecil
        noise_scale = rng.uniform(0.005, 0.02)
        base[:, 0] += rng.normal(0, noise_scale, size=WINDOW_SIZE)
        base[:, 1] += rng.normal(0, noise_scale, size=WINDOW_SIZE)

    elif attack_type == "velocity_drift":
        # Sensor drift gradual: 10-40 knots
        drift_amount = rng.uniform(10, 40) * rng.choice([-1, 1])
        drift = np.linspace(0, drift_amount, WINDOW_SIZE)
        base[:, 2] = np.clip(base[:, 2] + drift, 0, 350)

    elif attack_type == "dos_deletion":
        # Packet loss: set 2-3 timesteps ke nilai tidak wajar
        n_drop = rng.randint(2, 4)
        drop_idx = rng.choice(WINDOW_SIZE, size=n_drop, replace=False)
        for i in drop_idx:
            base[i, :] = 0.0  # semua fitur di-reset

    elif attack_type == "flight_merge" and donor_window is not None:
        # Trajectory takeover: replace 3-4 timestep terakhir
        donor_base = donor_window[:, :5].copy()
        n_replace = min(rng.randint(3, 5), WINDOW_SIZE - 1)
        base[-n_replace:, :] = donor_base[-n_replace:, :]
        # Noise transisi
        base[-n_replace-1:-n_replace+2, :2] += rng.normal(0, 0.1,
                                                          size=(3, 2))

    elif attack_type == "heading_manipulation":
        # Heading error: shift ±5/10/20 derajat di beberapa timestep
        n_shift = rng.randint(3, 8)
        shift_idx = rng.choice(WINDOW_SIZE, size=n_shift, replace=False)
        for i in shift_idx:
            offset = rng.choice([-20, -10, -5, 5, 10, 20])
            base[i, 4] = (base[i, 4] + offset) % 360

    return recompute_derived(base)


def main():
    log("=" * 60)
    log("REINJECT ATTACKS — Realistic Magnitude")
    log("=" * 60)

    rng = np.random.RandomState(RANDOM_SEED)

    # ── Load data ──
    log("\nLoad checkpoint data...")
    train_data = np.load(os.path.join(DATA_DIR, "train_data.npy")).astype(np.float32)
    val_data = np.load(os.path.join(DATA_DIR, "val_data.npy")).astype(np.float32)

    log(f"  Train: {train_data.shape} (100% normal)")
    log(f"  Val:   {val_data.shape}   (100% normal)")

    # ── Gabung train+val sebagai pool donor untuk flight_merge ──
    donor_pool = np.concatenate([train_data, val_data], axis=0)
    log(f"  Donor pool: {donor_pool.shape} windows")

    # ── Gunakan val_data sebagai basis test set baru ──
    # (val_data 100% normal, belum kena attack)
    n_test = len(val_data)
    test_data = val_data.copy()
    test_labels = np.zeros(n_test, dtype=np.int64)
    test_attack_types = np.full(n_test, "normal", dtype=object)

    log(f"\nTest set size: {n_test} windows")

    # ── Alokasi attack ──
    available_indices = list(range(n_test))
    rng.shuffle(available_indices)

    ptr = 0
    total_anom = 0
    for attack_name, fraction in ATTACK_FRACTIONS.items():
        n_attack = max(1, int(n_test * fraction))
        n_attack = min(n_attack, len(available_indices) - ptr)
        attack_idx = available_indices[ptr:ptr + n_attack]
        ptr += n_attack

        for idx in attack_idx:
            # Cari donor window untuk flight_merge (beda window)
            donor_idx = rng.randint(len(donor_pool))
            donor_window = donor_pool[donor_idx]

            # Inject attack
            modified = inject_attack(test_data[idx], attack_name, rng, donor_window)
            test_data[idx] = modified
            test_labels[idx] = 1
            test_attack_types[idx] = attack_name
            total_anom += 1

        log(f"  {attack_name:25s}: {n_attack:4d} windows terinfeksi")

    # ── Sisa windows tetap normal ──
    log(f"\n  Total anomali: {total_anom}/{n_test} "
        f"({100*total_anom/max(n_test,1):.1f}%)")
    log(f"  Total normal:  {n_test - total_anom}/{n_test} "
        f"({100*(n_test-total_anom)/max(n_test,1):.1f}%)")

    # ── Simpan ──
    log("\nSimpan ke checkpoint...")
    np.save(os.path.join(OUTPUT_DIR, "train_data.npy"), train_data)
    np.save(os.path.join(OUTPUT_DIR, "val_data.npy"), val_data)
    np.save(os.path.join(OUTPUT_DIR, "test_data.npy"), test_data)
    np.save(os.path.join(OUTPUT_DIR, "test_labels.npy"), test_labels)

    # test_attack_types: simpan sebagai array string
    np.save(os.path.join(OUTPUT_DIR, "test_attack_types.npy"),
            test_attack_types.astype(str))

    log(f"\n  train_data.npy        {train_data.shape}")
    log(f"  val_data.npy          {val_data.shape}")
    log(f"  test_data.npy         {test_data.shape}")
    log(f"  test_labels.npy       {test_labels.shape} "
        f"(anomaly: {100*test_labels.mean():.1f}%)")
    log(f"  test_attack_types.npy {test_attack_types.shape}")

    # ── Verifikasi ──
    log("\nVerifikasi distribusi attack:")
    unique, counts = np.unique(test_attack_types, return_counts=True)
    for u, c in zip(unique, counts):
        pct = 100 * c / n_test
        is_anom_counts = test_labels[test_attack_types == u].sum() if u != "normal" else 0
        if u == "normal":
            log(f"  {u:25s}: {c:4d} windows ({pct:.1f}%)  — no anomaly labels")
        else:
            log(f"  {u:25s}: {c:4d} windows ({pct:.1f}%)  — {is_anom_counts} labeled anomaly")

    log(f"\n{'=' * 60}")
    log("SELESAI!")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
