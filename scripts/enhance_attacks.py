"""
Patch: Perbesar magnitude attack pada test dataset yang sudah ada.
Menghindari kebutuhan untuk re-run Elasticsearch pipeline.
"""
import numpy as np
import os
import sys
from datetime import datetime

DATA_DIR = os.path.join("data", "checkpoint")
BACKUP_DIR = os.path.join("data", "checkpoint", "backup_pre_enhance")

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

def main():
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Load existing data
    test_data = np.load(os.path.join(DATA_DIR, "test_data.npy"))       # (1383, 10, 10)
    test_labels = np.load(os.path.join(DATA_DIR, "test_labels.npy"))   # (1383,)
    test_attack_types = np.load(os.path.join(DATA_DIR, "test_attack_types.npy"))

    log(f"Loaded test_data: {test_data.shape}")
    log(f"Loaded test_labels: {test_labels.shape} ({test_labels.mean()*100:.1f}% anomaly)")
    log(f"Attack types: {dict(zip(*np.unique(test_attack_types, return_counts=True)))}")

    # Backup original
    np.save(os.path.join(BACKUP_DIR, "test_data.npy"), test_data)
    np.save(os.path.join(BACKUP_DIR, "test_labels.npy"), test_labels)
    np.save(os.path.join(BACKUP_DIR, "test_attack_types.npy"), test_attack_types)
    log(f"Backup saved to {BACKUP_DIR}/")

    rng = np.random.RandomState(42)
    n_modified = 0

    for atk in ["constant_position", "velocity_drift", "flight_merge"]:
        mask = test_attack_types == atk
        indices = np.where(mask)[0]
        n = len(indices)
        if n == 0:
            continue
        log(f"Enhancing {atk}: {n} windows")

        for idx in indices:
            window = test_data[idx].copy()

            if atk == "constant_position":
                fake_lat = rng.uniform(-10, 10)
                fake_lon = rng.uniform(-10, 10)
                window[:, 0] = fake_lat
                window[:, 1] = fake_lon

            elif atk == "velocity_drift":
                n_quarter = max(2, 10 // 4)
                drift = np.zeros(10)
                drift[n_quarter:] = np.linspace(100, 300, 10 - n_quarter) * rng.choice([-1, 1])
                window[:, 2] = np.clip(window[:, 2] + drift, 0, 350)

            elif atk == "flight_merge":
                n_replace = 5
                replace_idx = rng.choice(10, n_replace, replace=False)
                for t in replace_idx:
                    window[t, 0] += rng.uniform(-3, 3)
                    window[t, 1] += rng.uniform(-3, 3)
                    window[t, 2] = rng.uniform(0, 350)
                    window[t, 3] += rng.uniform(-5000, 5000)
                # Transition noise
                for t in [4, 5]:
                    if t < 10:
                        window[t, 0] += rng.normal(0, 2)
                        window[t, 1] += rng.normal(0, 2)

            # Recompute derived features (dlat, dlon, dvel, dalt, dtrack)
            original = window[:, :5]
            deltas = np.diff(original, axis=0, prepend=original[0:1])
            window[:, 5:] = deltas

            test_data[idx] = window
            n_modified += 1

    log(f"Modified {n_modified} windows")

    # Update labels: pastikan label masih sesuai (yang dimodifikasi ya anomali)
    # Labels already 1 for these attacks, no change needed

    # Save
    np.save(os.path.join(DATA_DIR, "test_data.npy"), test_data)
    log(f"Saved enhanced test_data.npy")

    # Verify
    log(f"\nVerification:")
    log(f"  test_data: {test_data.shape}")
    log(f"  test_labels: {test_labels.mean()*100:.1f}% anomaly")
    log(f"  Attack rates:")
    for atk in np.unique(test_attack_types):
        if atk == "normal":
            continue
        mask = test_attack_types == atk
        log(f"    {atk:25s}: {mask.sum():4d} windows")

    log(f"\nDone! Original backed up at {BACKUP_DIR}/")

if __name__ == "__main__":
    main()
