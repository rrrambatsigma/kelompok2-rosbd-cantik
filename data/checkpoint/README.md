# Dataset Tahap 1 — Persiapan Data VAE-LSTM + SVDD

Dataset hasil dari `preprocessing/dump_data.py` siap digunakan untuk **Tahap 2 (Training)**.

---

## 📊 Ringkasan Dataset

| File | Shape | Ukuran | Isi |
|---|---|---|---|
| `train_data.npy` | (6.591, 10, 10) | 2.5 MB | **100% normal** — 350 flight |
| `val_data.npy` | (1.409, 10, 10) | 539 KB | **100% normal** — 75 flight |
| `test_data.npy` | (1.383, 10, 10) | 530 KB | Normal + 6 jenis anomali |
| `test_labels.npy` | (1.383,) | 11 KB | 0=normal, 1=anomali (26.2% anomali) |
| `test_attack_types.npy` | (1.383,) | 109 KB | Label jenis anomali per window |

### Total

| Metrik | Nilai |
|---|---|
| Total windows | 9.383 |
| Total record (10× window) | 93.830 |
| Train / Val / Test | 70% / 15% / 15% |

---

## ⚙️ Cara Regenerate

```bash
python preprocessing/dump_data.py
```

Butuh ~1 menit untuk query + proses 622.529 records dari Elasticsearch.

---

## 📐 Konfigurasi

| Parameter | Nilai |
|---|---|
| **Sumber data** | `flights-remote` (Elasticsearch local, hasil reindex dari 100.99.130.69) |
| **Rentang waktu** | 14 hari terakhir |
| **Fitur original (5)** | `latitude`, `longitude`, `velocity`, `baro_altitude`, `true_track` |
| **Fitur derived (5)** | `dlat`, `dlon`, `dvel`, `dalt`, `dtrack` — delta antar-timestep |
| **Total fitur** | **10** (original + derived) |
| **Window size** | 10 (≈ 5 menit per window) |
| **Stride** | 5 (overlap 50%) |
| **Segmentasi flight** | Gap timestamp > 30 menit = flight baru |
| **Min flight length** | 10 record |
| **Stratified sampling** | 500 flight per region (hanya Europe) |
| **Split** | Per flight_id (70/15/15) — **tanpa data leakage** |
| **6 jenis anomali** | Diinjeksikan ke test set HANYA (train/val 100% normal) |

---

## ✅ Verifikasi

- [x] **Tidak ada data leakage** — flight_id unik di setiap split
- [x] **Train 100% normal** — tidak terkontaminasi anomali
- [x] **Val 100% normal** — threshold jujur dari data bersih
- [x] **6 jenis anomali di test** — distribusi sesuai proporsi
- [x] **Window shape benar** — (N, 10, 10)

### Distribusi Anomali di Test Set

| Jenis | Jumlah | Persentase Test | Kategori |
|---|---|---|---|
| normal | 1.021 | 73.8% | NORMAL |
| constant_position | 134 | 9.7% | ANOMALI |
| velocity_drift | 77 | 5.6% | ANOMALI |
| random_position | 59 | 4.3% | ANOMALI |
| heading_manipulation | 46 | 3.3% | ANOMALI |
| flight_merge | 37 | 2.7% | ANOMALI |
| dos_deletion | 9 | 0.7% | ANOMALI |
| **Total** | **1.383** | **100%** | |

---

## 🎯 Hasil Training (Reconstruction-only Scoring)

| Metrik | Nilai |
|---|---|
| **AUC-ROC** | **0.643** |
| **Anomaly detection** | **125/362 TP, 40 FP** |

### Deteksi Per-Jenis Anomali

| Jenis Anomali | Deteksi | Keterangan |
|---|---|---|
| random_position | **59/59 (100%)** | Sempurna — jitter terdeteksi dari delta fitur |
| heading_manipulation | **44/46 (96%)** | Hampir sempurna — perubahan arah mendadak |
| dos_deletion | **6/9 (67%)** | Baik — gap data terdeteksi |
| velocity_drift | **8/77 (10%)** | Sebagian — drift gradual masih rapuh |
| flight_merge | **5/37 (14%)** | Sebagian — perlu donor sangat berbeda |
| constant_position | **3/134 (2%)** | **Tidak terdeteksi** — offset mulus tidak bisa dideteksi VAE |

> **Catatan:** `constant_position` secara fundamental tidak bisa dideteksi VAE karena pergeseran posisi tetap menghasilkan trayektori yang mulus. Deteksi jenis ini membutuhkan konteks posisi absolut (jalur penerbangan, bandara, dll).

---

## 🔗 Hubungan ke Tahap 2

File-file ini dipakai oleh `modelling/train.py`:

```
data/checkpoint/
├── train_data.npy      → train VAE-LSTM + SVDD
├── val_data.npy        → hitung threshold (persentil 95)
├── test_data.npy       → evaluasi final
├── test_labels.npy     → ground truth untuk metrik
└── test_attack_types.npy → evaluasi per-jenis anomali
```

**Input ke model:** (batch, 10, 10)
**Target output:** anomaly_score (float) + is_anomaly (bool)

### Model yang Dihasilkan (Tahap 2)

```
models/vae-svdd/
├── vae_model.pt         ← VAE-LSTM (encoder LSTM + decoder LSTM)
├── svdd_model.pkl       ← OneClassSVM di latent z (opsional)
├── scaler.pkl           ← StandardScaler (fit dari train)
├── config.json          ← threshold, feature_names, dll
└── metrics.json         ← hasil evaluasi
```
