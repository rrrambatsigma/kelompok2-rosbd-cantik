"""
dump_data.py — Tahap 1: Persiapan Data untuk VAE-LSTM + SVDD

ALUR LENGKAP:
  1. Query Elasticsearch (14 hari terakhir via scroll)
  2. Cleaning data (buang on_ground, outlier, null)
  3. Segmentasi per penerbangan (gap timestamp > 1800 detik)
  4. Buang penerbangan terlalu pendek (< 10 record)
  5. Stratified sampling per region (~150 flight/region)
  6. Split per flight_id (70/15/15) — TANPA DATA LEAKAGE
  7. Inject 6 jenis anomali ke test set
  8. Sliding window (size=10, stride=5)
  9. Simpan file .npy ke data/checkpoint/

PRINSIP:
  - Train/Val = 100% data NORMAL
  - Test = campuran normal + anomali (dengan label)
  - Split per FLIGHT, bukan per RECORD
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
from typing import Optional, Dict, Tuple

# ─── KONFIGURASI ────────────────────────────────────────────
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "flights-remote")      # alias hasil reindex
OUTPUT_DIR = os.path.join("data", "checkpoint")

FEATURES = [
    "latitude", "longitude", "velocity",
    "baro_altitude", "true_track"
]

# Fitur derived — perubahan antar-timestep untuk menangkap dinamika
DERIVED_FEATURES = [
    "dlat", "dlon", "dvel", "dalt", "dtrack"
]

ALL_FEATURES = FEATURES + DERIVED_FEATURES

WINDOW_SIZE = 10
STRIDE = 5
SEGMENT_GAP = 1800     # 30 menit dalam detik
MIN_FLIGHT_LEN = 10    # minimal record per flight
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

N_FLIGHT_PER_REGION = 3000

RANDOM_SEED = 42

ATTACK_FRACTIONS = {
    "constant_position": 0.08,
    "random_position": 0.08,
    "velocity_drift": 0.06,
    "dos_deletion": 0.05,
    "flight_merge": 0.05,
    "heading_manipulation": 0.06,
}

# ─── LOGGING ────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


# ─── LANGKAH 1: FETCH DATA ─────────────────────────────────
def fetch_data(es_host: str = ES_HOST, index: str = ES_INDEX,
               days: int = 14, size: int = 10000) -> pd.DataFrame:
    """Query Elasticsearch dengan scroll API — 14 hari terakhir."""
    es_url = es_host if es_host.startswith("http") else f"http://{es_host}"
    es = Elasticsearch(es_url, request_timeout=120, max_retries=3, retry_on_timeout=True)

    log(f"Query {days} hari terakhir dari index '{index}'...")

    # Hitung epoch 14 hari lalu (karena timestamp bertipe float, bukan date)
    fourteen_days_ago = time.time() - days * 24 * 3600

    result = es.search(
        index=index,
        scroll="10m",
        size=size,
        body={
            "query": {
                "range": {
                    "timestamp": {
                        "gte": fourteen_days_ago
                    }
                }
            },
            "sort": [{"timestamp": {"order": "asc"}}],
            "_source": [
                "icao24", "callsign", "timestamp",
                "latitude", "longitude", "velocity",
                "baro_altitude", "true_track",
                "region", "on_ground"
            ]
        }
    )

    all_docs = []
    scroll_id = result.get("_scroll_id")
    hits = result["hits"]["hits"]
    all_docs.extend([h["_source"] for h in hits])
    log(f"  Batch 1: {len(hits)} docs")

    batch = 2
    while len(hits) > 0:
        try:
            result = es.scroll(scroll_id=scroll_id, scroll="10m", request_timeout=60)
            scroll_id = result.get("_scroll_id")
            hits = result["hits"]["hits"]
            all_docs.extend([h["_source"] for h in hits])
            if hits:
                log(f"  Batch {batch}: {len(hits)} docs (total: {len(all_docs)})")
                batch += 1
        except Exception as e:
            log(f"  Scroll error: {e}, retry dengan scroll baru...")
            result = es.search(
                index=index,
                scroll="10m",
                size=size,
                body={
                    "query": {
                        "range": {
                            "timestamp": {
                                "gte": fourteen_days_ago
                            }
                        }
                    },
                    "sort": [{"timestamp": {"order": "asc"}}],
                    "_source": [
                        "icao24", "callsign", "timestamp",
                        "latitude", "longitude", "velocity",
                        "baro_altitude", "true_track",
                        "region", "on_ground"
                    ]
                }
            )
            scroll_id = result.get("_scroll_id")
            hits = result["hits"]["hits"]
            if hits:
                log(f"  Restart scroll: {len(hits)} docs (total: {len(all_docs)})")

    es.clear_scroll(scroll_id=scroll_id)

    df = pd.DataFrame(all_docs)
    log(f"  TOTAL: {len(df)} records dari ES")
    return df


# ─── LANGKAH 2: CLEANING ───────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Buang data tidak layak."""
    n0 = len(df)
    log("Cleaning data...")

    # Buang yang tidak punya posisi
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    log(f"  Buang null position: {before} -> {len(df)}")

    # Buang pesawat di darat
    if "on_ground" in df.columns:
        before = len(df)
        df = df[df["on_ground"] == False]
        log(f"  Buang on_ground: {before} -> {len(df)}")

    # Filter velocity
    if "velocity" in df.columns:
        before = len(df)
        df = df[df["velocity"].between(0, 350)]
        log(f"  Filter velocity [0,350]: {before} -> {len(df)}")

    # Filter altitude (baro_altitude khas pesawat komersial)
    if "baro_altitude" in df.columns:
        before = len(df)
        df = df[df["baro_altitude"].between(-500, 20000)]
        log(f"  Filter baro_altitude [-500,20000]: {before} -> {len(df)}")

    # Filter true_track
    if "true_track" in df.columns:
        before = len(df)
        df = df[df["true_track"].between(0, 360)]
        log(f"  Filter true_track [0,360]: {before} -> {len(df)}")

    # Buang yang masih null di fitur utama
    before = len(df)
    df = df.dropna(subset=FEATURES)
    log(f"  Buang null fitur: {before} -> {len(df)}")

    # Drop duplicates per (icao24, timestamp)
    before = len(df)
    df = df.drop_duplicates(subset=["icao24", "timestamp"])
    log(f"  Buang duplikat: {before} -> {len(df)}")

    log(f"  HASIL CLEANING: {len(df)}/{n0} tersisa ({100*len(df)/max(n0,1):.1f}%)")
    return df.reset_index(drop=True)


# ─── LANGKAH 3: SEGMENTASI FLIGHT ──────────────────────────
def segment_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Kelompokkan per icao24, potong jika gap > 30 menit."""
    log("Segmentasi flight (gap > 30 menit)...")
    df = df.sort_values(["icao24", "timestamp"]).reset_index(drop=True)

    # Hitung gap timestamp per icao24
    df["time_diff"] = df.groupby("icao24")["timestamp"].diff()

    # Tandai awal flight baru
    df["new_flight"] = df["time_diff"] > SEGMENT_GAP

    # Flight ID: icao24 + nomor urut
    df["flight_id"] = df.groupby("icao24")["new_flight"].cumsum()
    df["flight_id"] = df["icao24"] + "_" + df["flight_id"].astype(int).astype(str)

    n_flights = df["flight_id"].nunique()
    n_icao = df["icao24"].nunique()
    log(f"  {n_icao} pesawat -> {n_flights} segmen flight")
    return df


# ─── LANGKAH 4: BUANG FLIGHT PENDEK ────────────────────────
def filter_short_flights(df: pd.DataFrame, min_len: int = MIN_FLIGHT_LEN) -> pd.DataFrame:
    """Buang flight dengan record < min_len."""
    before = len(df)
    flight_counts = df["flight_id"].value_counts()
    valid = flight_counts[flight_counts >= min_len].index
    df = df[df["flight_id"].isin(valid)]
    n_buang = before - len(df)
    n_flight_buang = len(flight_counts) - len(valid)
    log(f"  Buang {n_flight_buang} flight (< {min_len} record): {before} -> {len(df)}")
    return df.reset_index(drop=True)


# ─── LANGKAH 5: STRATIFIED SAMPLING ────────────────────────
def stratified_sample(df: pd.DataFrame,
                      n_per_region: int = N_FLIGHT_PER_REGION,
                      seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Ambil ~n_per_region flight per region."""
    rng = np.random.RandomState(seed)
    regions = df["region"].unique() if "region" in df.columns else ["unknown"]
    log(f"  Region ditemukan: {list(regions)}")

    sampled = []
    for region in regions:
        if region == "unknown":
            region_df = df
        else:
            region_df = df[df["region"] == region]

        flight_ids = region_df["flight_id"].unique()

        if len(flight_ids) <= n_per_region:
            log(f"    {region}: {len(flight_ids)} flight (ambil semua)")
            sampled.append(region_df)
        else:
            chosen = rng.choice(flight_ids, size=n_per_region, replace=False)
            sampled.append(region_df[region_df["flight_id"].isin(chosen)])
            log(f"    {region}: sample {n_per_region}/{len(flight_ids)} flight")

    result = pd.concat(sampled, ignore_index=True)
    log(f"  HASIL SAMPLING: {result['flight_id'].nunique()} flight, {len(result)} records")
    return result


# ─── LANGKAH 6: SPLIT PER FLIGHT ──────────────────────────
def split_by_flight(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split 70/15/15 per flight_id — TANPA DATA LEAKAGE."""
    flight_ids = df["flight_id"].unique()
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(flight_ids)

    n_total = len(flight_ids)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_flights = flight_ids[:n_train]
    val_flights = flight_ids[n_train:n_train + n_val]
    test_flights = flight_ids[n_train + n_val:]

    train_df = df[df["flight_id"].isin(train_flights)].copy()
    val_df = df[df["flight_id"].isin(val_flights)].copy()
    test_df = df[df["flight_id"].isin(test_flights)].copy()

    log(f"  Split: {len(train_flights)} train / {len(val_flights)} val"
        f" / {len(test_flights)} test flight")
    log(f"  Records: {len(train_df)} / {len(val_df)} / {len(test_df)}")

    return train_df, val_df, test_df


# ─── LANGKAH 7: INJECT ANOMALI ─────────────────────────────
def inject_anomalies(df: pd.DataFrame,
                     fractions: Optional[Dict[str, float]] = None,
                     seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Inject 6 jenis anomali ke test set.
    HANYA memodifikasi data, TIDAK membuat flight baru.
    """
    if fractions is None:
        fractions = ATTACK_FRACTIONS
    rng = np.random.RandomState(seed)

    # Pastikan kolom numerik
    df = df.copy()
    for c in FEATURES:
        if c in df.columns:
            df[c] = df[c].astype(float)

    df["is_anomaly"] = False
    df["attack_type"] = None

    # Daftar flight yang tersedia
    available_flights = list(df["flight_id"].unique())
    used_flights = set()  # flight yang sudah kena anomali

    for attack_name, fraction in fractions.items():
        # Flight yang belum kena anomali
        candidates = [f for f in available_flights if f not in used_flights]
        if not candidates:
            break

        n_select = max(1, int(len(candidates) * fraction))
        chosen_flights = rng.choice(candidates, size=min(len(candidates), n_select),
                                    replace=False)

        for fid in chosen_flights:
            used_flights.add(fid)
            mask = df["flight_id"] == fid
            idx = df.index[mask]
            n = len(idx)
            if n == 0:
                continue

            # Tandai sebagai anomali
            df.loc[idx, "is_anomaly"] = True
            df.loc[idx, "attack_type"] = attack_name

            # Terapkan serangan sesuai jenis — realistic magnitude
            if attack_name == "constant_position":
                # GPS spoofing: posisi stuck di titik pertama (sensor macet)
                first_lat = df.loc[idx[0], "latitude"]
                first_lon = df.loc[idx[0], "longitude"]
                df.loc[idx, "latitude"] = first_lat + rng.normal(0, 0.005, size=n)
                df.loc[idx, "longitude"] = first_lon + rng.normal(0, 0.005, size=n)

            elif attack_name == "random_position":
                # GPS jitter: noise posisi kecil ~500m-2km
                noise_scale = rng.uniform(0.005, 0.02)
                df.loc[idx, "latitude"] = (df.loc[idx, "latitude"].values +
                                            rng.normal(0, noise_scale, size=n))
                df.loc[idx, "longitude"] = (df.loc[idx, "longitude"].values +
                                             rng.normal(0, noise_scale, size=n))

            elif attack_name == "velocity_drift":
                # Sensor drift gradual: 10-40 knots drift
                base = df.loc[idx, "velocity"].values.astype(float)
                n_quarter = max(2, n // 4)
                drift = np.zeros(n)
                drift_amount = rng.uniform(10, 40)
                drift[n_quarter:] = np.linspace(0, drift_amount, n - n_quarter) * rng.choice([-1, 1])
                df.loc[idx, "velocity"] = np.clip(base + drift, 0, 350)

            elif attack_name == "dos_deletion":
                # Packet loss: hapus 20-30% data
                drop_ratio = rng.uniform(0.2, 0.3)
                n_keep = max(1, int(n * (1 - drop_ratio)))
                keep_idx = rng.choice(idx, size=n_keep, replace=False)
                drop_idx = idx.difference(keep_idx)
                df = df.drop(drop_idx)

            elif attack_name == "flight_merge":
                # Trajectory takeover: 30% trajectory diganti
                donor_pool = [f for f in available_flights
                              if f not in used_flights and f != fid]
                if donor_pool:
                    donor_fid = rng.choice(donor_pool)
                    donor_data = df[df["flight_id"] == donor_fid]
                    if len(donor_data) > 0:
                        n_replace = min(len(donor_data), max(3, int(n * 0.3)))
                        replace_idx = idx[-n_replace:]
                        donor_vals = donor_data.iloc[-n_replace:]
                        for col in ["latitude", "longitude",
                                    "velocity", "baro_altitude"]:
                            vals = donor_vals[col].values
                            if len(vals) > 0:
                                df.loc[replace_idx[:len(vals)], col] = vals
                    # Noise kecil di titik transisi
                    transition = idx[max(0, len(idx)//2 - 1):min(len(idx), len(idx)//2 + 2)]
                    if len(transition) > 0:
                        noise = rng.normal(0, 0.1, size=(len(transition),))
                        df.loc[transition, "latitude"] = df.loc[transition, "latitude"].values + noise
                        df.loc[transition, "longitude"] = df.loc[transition, "longitude"].values + noise

            elif attack_name == "heading_manipulation":
                # Heading error: shift ±5°/±10°/±20° (error sensor)
                offset = rng.choice([-20, -10, -5, 5, 10, 20], size=n)
                df.loc[idx, "true_track"] = (
                    df.loc[idx, "true_track"].values + offset
                ) % 360

        log(f"  {attack_name}: {len(chosen_flights)} flight terinfeksi")

    df = df.reset_index(drop=True)
    n_anomali = df["is_anomaly"].sum()
    log(f"  TOTAL ANOMALI: {n_anomali}/{len(df)} "
        f"({100*n_anomali/max(len(df),1):.1f}%)")
    return df


# ─── LANGKAH 8: DERIVED FEATURES ──────────────────────────
def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung delta features (perubahan antar-timestep) per flight.
    Delta features menangkap dinamika penerbangan, bukan posisi absolut.
    """
    log("Menghitung derived features (delta antar-timestep)...")
    df = df.sort_values(["flight_id", "timestamp"]).reset_index(drop=True)

    df["dlat"] = df.groupby("flight_id")["latitude"].diff().fillna(0)
    df["dlon"] = df.groupby("flight_id")["longitude"].diff().fillna(0)
    df["dvel"] = df.groupby("flight_id")["velocity"].diff().fillna(0)
    df["dalt"] = df.groupby("flight_id")["baro_altitude"].diff().fillna(0)
    df["dtrack"] = df.groupby("flight_id")["true_track"].diff().fillna(0)

    log(f"  Derived features ditambahkan: "
        f"{['dlat','dlon','dvel','dalt','dtrack']}")
    return df


# ─── LANGKAH 9: SLIDING WINDOW ─────────────────────────────
def sliding_windows(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Buat sliding window dari setiap flight.
    Return: (windows, labels, flight_ids)
    - windows.shape = (N, WINDOW_SIZE, n_features)
    - labels.shape = (N,)  — 0=normal, 1=anomaly
    - flight_ids.shape = (N,)
    """
    all_windows = []
    all_labels = []
    all_flight_ids = []

    has_label = "is_anomaly" in df.columns
    feature_cols = ALL_FEATURES  # original + derived

    flight_ids = df["flight_id"].unique()
    log(f"  Membuat window dari {len(flight_ids)} flight...")
    log(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    for fid in flight_ids:
        fdf = df[df["flight_id"] == fid].sort_values("timestamp")
        values = fdf[feature_cols].values.astype(np.float32)

        for i in range(0, len(values) - WINDOW_SIZE + 1, STRIDE):
            window = values[i:i + WINDOW_SIZE]
            all_windows.append(window)

            if has_label:
                # Jika > 50% record dalam window adalah anomali → label 1
                labels_in_window = fdf["is_anomaly"].values[i:i + WINDOW_SIZE]
                is_anom = labels_in_window.astype(int).mean() > 0.5
                all_labels.append(int(is_anom))
            else:
                all_labels.append(0)

            all_flight_ids.append(fid)

    if not all_windows:
        log(f"  PERINGATAN: Tidak ada window yang dihasilkan!")
        return np.array([]).reshape(0, WINDOW_SIZE, len(FEATURES)), \
               np.array([]), np.array([])

    windows = np.array(all_windows)
    labels = np.array(all_labels)
    flight_ids_arr = np.array(all_flight_ids, dtype=object)

    log(f"  HASIL: {len(windows)} window, shape={windows.shape}")
    return windows, labels, flight_ids_arr


# ─── LANGKAH 9: SIMPAN ─────────────────────────────────────
def save_dataset(train_data, val_data, test_data, test_labels,
                 test_flight_ids, test_attack_types, output_dir: str):
    """Simpan semua file .npy ke output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "train_data.npy"), train_data)
    log(f"  Saved: train_data.npy {train_data.shape}")

    np.save(os.path.join(output_dir, "val_data.npy"), val_data)
    log(f"  Saved: val_data.npy {val_data.shape}")

    np.save(os.path.join(output_dir, "test_data.npy"), test_data)
    log(f"  Saved: test_data.npy {test_data.shape}")

    np.save(os.path.join(output_dir, "test_labels.npy"), test_labels)
    anom_rate = test_labels.mean()
    log(f"  Saved: test_labels.npy ({len(test_labels)} labels, "
        f"{100*anom_rate:.1f}% anomali)")

    if test_attack_types is not None:
        np.save(os.path.join(output_dir, "test_attack_types.npy"),
                test_attack_types)
        log(f"  Saved: test_attack_types.npy")

    log(f"\n  Semua file tersimpan di: {output_dir}/")


# ─── MAIN ──────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("TAHAP 1: DUMP DATA — Persiapan Dataset")
    log("=" * 60)

    t_start = time.time()

    # ── Langkah 1: Fetch ──
    log("\n[1/9] Fetch data dari Elasticsearch...")
    raw_df = fetch_data()

    # ── Langkah 2: Cleaning ──
    log("\n[2/9] Cleaning data...")
    clean_df = clean_data(raw_df)

    # ── Langkah 3: Segmentasi ──
    log("\n[3/9] Segmentasi flight...")
    seg_df = segment_flights(clean_df)

    # ── Langkah 4: Buang flight pendek ──
    log("\n[4/9] Filter flight pendek...")
    seg_df = filter_short_flights(seg_df)

    # ── Langkah 5: Stratified sampling ──
    log("\n[5/9] Stratified sampling...")
    sampled_df = stratified_sample(seg_df)

    # ── Langkah 6: Split ──
    log("\n[6/9] Split train/val/test per flight_id...")
    train_df, val_df, test_df = split_by_flight(sampled_df)

    # Pastikan train/val 100% normal (belum ada anomali)
    log("  Train: 100% normal")
    log("  Val:   100% normal")

    # ── Langkah 7: Inject anomali ke test ──
    log("\n[7/9] Inject 6 jenis anomali ke test set...")
    test_df = inject_anomalies(test_df)
    log(f"  Test: {test_df['is_anomaly'].sum()} anomali / {len(test_df)} total")

    # ── Langkah 8: Derived features ──
    log("\n[8/10] Compute derived features (delta antar-timestep)...")
    train_df = compute_derived_features(train_df)
    val_df = compute_derived_features(val_df)
    test_df = compute_derived_features(test_df)

    # ── Langkah 9: Sliding window ──
    log("\n[9/10] Sliding window...")
    train_windows, _, _ = sliding_windows(train_df)
    val_windows, _, _ = sliding_windows(val_df)
    test_windows, test_labels, test_fids = sliding_windows(test_df)

    # Simpan attack types untuk test (untuk evaluasi per-jenis nanti)
    test_attack_map = {}
    if "attack_type" in test_df.columns:
        for fid in test_df["flight_id"].unique():
            types = test_df[test_df["flight_id"] == fid]["attack_type"].dropna()
            if len(types) > 0:
                test_attack_map[fid] = types.iloc[0]

    test_attack_arr = np.array([test_attack_map.get(fid, "normal")
                                for fid in test_fids])

    # ── Verifikasi ──
    log("\n[VERIFIKASI]")
    log(f"  Train windows:  {train_windows.shape}")
    log(f"  Val windows:    {val_windows.shape}")
    log(f"  Test windows:   {test_windows.shape}")
    log(f"  Test labels:    {test_labels.shape} "
        f"(anomaly: {100*test_labels.mean():.1f}%)")

    # Cek data leakage
    train_fids = set(train_df["flight_id"].unique())
    val_fids = set(val_df["flight_id"].unique())
    test_fids_set = set(test_df["flight_id"].unique())
    leakage = train_fids & test_fids_set
    if leakage:
        log(f"  [WARNING] DATA LEAKAGE TERDETEKSI! {len(leakage)} flight "
            f"ada di train dan test!")
    else:
        log(f"  [OK] Tidak ada data leakage (train n test = kosong)")

    leakage_vt = val_fids & test_fids_set
    if leakage_vt:
        log(f"  [WARNING] DATA LEAKAGE (val n test): {len(leakage_vt)} flight")
    else:
        log(f"  [OK] Tidak ada data leakage (val n test = kosong)")

    # ── Langkah 9: Simpan ──
    log("\n[10/10] Simpan dataset...")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", OUTPUT_DIR)
    save_dataset(train_windows, val_windows, test_windows, test_labels,
                 np.array(test_fids), test_attack_arr, output_dir)

    t_elapsed = time.time() - t_start
    log(f"\n{'=' * 60}")
    log(f"SELESAI! Waktu: {t_elapsed/60:.1f} menit")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
