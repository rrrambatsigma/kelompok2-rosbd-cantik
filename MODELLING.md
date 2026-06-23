# Panduan Modelling — VAE-LSTM + SVDD

Panduan ini khusus untuk **bagian modelling** deteksi anomali ADS-B. Baca urut dari atas ke bawah, jangan loncat-loncat.

---

## 📌 Ringkasan Model

| Komponen | Fungsi |
|---|---|
| **VAE-LSTM** | Belajar pola pergerakan pesawat normal. Seperti mesin fotokopi yang hafal gerakan normal. |
| **OneClassSVM** | Gambar lingkaran pengaman di sekeliling pola normal. Yang keluar lingkaran = anomali. |
| **Threshold** | Batas skor anomali. Dari data validasi, ambil persentil 95. |

### Input Model

```
1 window = 10 titik × 5 fitur = 50 angka
Fitur: latitude, longitude, velocity, baro_altitude, true_track
```

### Output Model

```
Untuk setiap window:
  - is_anomaly: true/false
  - anomaly_score: angka (makin besar makin anomali)
  - attack_type: jenis anomali (jika terdeteksi)
```

---

## ⚠️ Aturan Emas — Jangan Sampai Salah

### 1. Data Leakage (Kebocoran Data)

❌ **SALAH:** Split data secara acak per record
✅ **BENAR:** Split per flight_id

*Kenapa?* Kalau split acak, data pesawat yang sama bisa masuk train DAN test. Model jadi "curang" karena sudah lihat sebagian data pesawat itu. Skor evaluasi jadi palsu tinggi.

```
❌  Train: [flight_A_t0, flight_A_t1, flight_B_t0]
    Test:  [flight_A_t2, flight_B_t1, flight_B_t2]  ← bocor!
    
✅  Train: [flight_A_t0..t10, flight_C_t0..t8]
    Test:  [flight_B_t0..t12, flight_D_t0..t5]       ← aman
```

### 2. Anomali Tidak Boleh Masuk Training

❌ **SALAH:** Train pakai data campuran normal + anomali
✅ **BENAR:** Train 100% data normal. Anomali hanya di test.

*Kenapa?* Model harus belajar "wajah normal" dulu. Kalau training sudah ada anomali, model bingung mana yang normal.

### 3. Scaling Harus Konsisten

❌ **SALAH:** Fit scaler ulang setiap kali predict
✅ **BENAR:** Fit scaler SEKALI di train, simpan, pakai lagi di inference

*Kenapa?* Skala fitur harus sama antara training dan prediksi. Kalau berbeda, angka masuk ke model jadi kacau.

---

## 🗺️ Alur Modelling — 3 Tahap

```
┌─────────────────────────────────────────────────────────┐
│                   TAHAP 1: PERSIAPAN DATA                │
│  dump_data.py (jalan SEKALI)                             │
├─────────────────────────────────────────────────────────┤
│  ES → query → cleaning → segmentasi → split → window    │
│  → train.npy, val.npy, test_normal.npy, test_anomali.npy│
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   TAHAP 2: TRAINING                      │
│  train.py (jalan SEKALI)                                 │
├─────────────────────────────────────────────────────────┤
│  scaler → train VAE-LSTM → train SVDD → threshold       │
│  → vae.pt, svdd.pkl, scaler.pkl, config.json            │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   TAHAP 3: INFERENCE                      │
│  serving/api.py (jalan TERUS-MENERUS)                    │
├─────────────────────────────────────────────────────────┤
│  buffer per pesawat → scale → encode → score → SSE      │
└─────────────────────────────────────────────────────────┘
```

---

## 🥇 TAHAP 1: Persiapan Data (`dump_data.py`)

### Tujuan
Mengambil data dari Elasticsearch, membersihkan, dan menyiapkan windows untuk training.

### Langkah-langkah

```
Urutan:
  1. Query ES 14 hari terakhir (scroll API)
  2. Bersihkan data
  3. Kelompokkan per pesawat (icao24)
  4. Potong-potong jadi flight segments
  5. Buang flight yang terlalu pendek
  6. Sampling seimbang per region
  7. Split per flight_id: train 70%, val 15%, test 15%
  8. Sliding window (10, stride 5)
  9. Inject 6 jenis anomali ke test set
  10. Simpan .npy
```

### Detail Setiap Langkah

#### Langkah 1: Query ES
```python
# Ambil 14 hari terakhir
body = {
    "query": {
        "range": {
            "timestamp": {"gte": "now-14d"}
        }
    }
}
```

#### Langkah 2: Cleaning
```python
# Buang yang tidak layak
df = df[df["on_ground"] == False]
df = df[df["velocity"].between(0, 350)]
df = df[df["baro_altitude"].between(-500, 20000)]
df = df[df["true_track"].between(0, 360)]
```

#### Langkah 3-4: Segmentasi Flight
```python
# Gap > 30 menit = flight baru
df["time_diff"] = df.groupby("icao24")["timestamp"].diff()
df["new_flight"] = df["time_diff"] > 1800
df["flight_id"] = df.groupby("icao24")["new_flight"].cumsum()
df["flight_id"] = df["icao24"] + "_" + df["flight_id"].astype(str)
```

#### Langkah 5: Buang Flight Pendek
```python
# Minimal 10 record (biar bisa jadi 1 window)
flight_counts = df["flight_id"].value_counts()
valid_flights = flight_counts[flight_counts >= 10].index
df = df[df["flight_id"].isin(valid_flights)]
```

#### Langkah 7: Split per Flight (PENTING!)
```python
# Split per flight_id, BUKAN per record!
all_flights = df["flight_id"].unique()
np.random.shuffle(all_flights)

n_train = int(0.7 * len(all_flights))
n_val   = int(0.15 * len(all_flights))

train_flights = all_flights[:n_train]
val_flights   = all_flights[n_train:n_train + n_val]
test_flights  = all_flights[n_train + n_val:]

train_df = df[df["flight_id"].isin(train_flights)]
val_df   = df[df["flight_id"].isin(val_flights)]
test_df  = df[df["flight_id"].isin(test_flights)]
```

#### Langkah 8: Sliding Window
```python
WINDOW_SIZE = 10
STRIDE = 5

def sliding_window(flight_df):
    windows = []
    values = flight_df[FEATURES].values
    for i in range(0, len(values) - WINDOW_SIZE + 1, STRIDE):
        windows.append(values[i:i + WINDOW_SIZE])
    return np.array(windows)

# Apply ke setiap flight
train_windows = []
for fid in train_flights:
    fdf = train_df[train_df["flight_id"] == fid]
    w = sliding_window(fdf)
    if len(w) > 0:
        train_windows.append(w)
train_data = np.vstack(train_windows) if train_windows else np.array([])
```

#### Langkah 9: Inject Anomali ke Test
```python
# 6 jenis anomali dengan proporsi tertentu
fractions = {
    "constant_position": 0.08,
    "random_position": 0.08, 
    "velocity_drift": 0.06,
    "dos_deletion": 0.05,
    "flight_merge": 0.05,
    "heading_manipulation": 0.06,
}
```

### ✅ Verifikasi Tahap 1

```bash
# Cek jumlah file
ls -lh data/checkpoint/

# Output yang diharapkan:
# train_data.npy       → ~XX MB (normal, untuk training)
# val_data.npy         → ~XX MB (normal, untuk threshold)
# test_normal.npy      → ~XX MB (normal, untuk evaluasi)
# test_anomaly.npy     → ~XX MB (campuran normal+anomali)
# test_labels.npy      → label 0/1 untuk evaluasi

# Cek bentuk array
python -c "
import numpy as np
t = np.load('data/checkpoint/train_data.npy')
print(f'Train shape: {t.shape}')   # (n_windows, 10, 5)
print(f'Range: {t.min():.2f} - {t.max():.2f}')
"
```

### ❌ Common Mistakes Tahap 1

| Masalah | Akibat | Solusi |
|---|---|---|
| Split per record | Data leakage, evaluasi palsu | Split per flight_id |
| Lupa buang on_ground | Data pesawat di darat ikut belajar | Filter on_ground=False |
| Flight pendek ikut | Window tidak penuh | Buang < 10 record |
| Tidak sampling region | Dominasi region tertentu | Sample ~150 flight/region |

---

## 🥈 TAHAP 2: Training (`modelling/train.py`)

### Tujuan
Melatih VAE-LSTM + OneClassSVM, menentukan threshold, dan evaluasi.

### Langkah-langkah

```
Urutan:
  1. Load data .npy
  2. Fit scaler di DATA TRAIN SAJA
  3. Train VAE-LSTM
  4. Extract latent dari train
  5. Train OneClassSVM di latent
  6. Hitung threshold dari VAL set
  7. Evaluasi di test set (normal + anomali)
  8. Simpan semua artifacts
```

### Detail Setiap Langkah

#### Langkah 1: Load Data
```python
X_train = np.load("data/checkpoint/train_data.npy")   # (N, 10, 5)
X_val   = np.load("data/checkpoint/val_data.npy")       # (N, 10, 5)
X_test  = np.load("data/checkpoint/test_anomaly.npy")   # (N, 10, 5)
y_test  = np.load("data/checkpoint/test_labels.npy")    # (N,) 0/1
```

#### Langkah 2: Scaler
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit hanya dari train data
# Reshape: (N, 10, 5) → (N*10, 5) → fit → reshape balik
N_train, T, F = X_train.shape
X_train_2d = X_train.reshape(-1, F)
scaler.fit(X_train_2d)

# Transform semua set (train, val, test)
X_train_scaled = scaler.transform(X_train_2d).reshape(-1, T, F)
X_val_scaled   = scaler.transform(X_val.reshape(-1, F)).reshape(-1, T, F)
X_test_scaled  = scaler.transform(X_test.reshape(-1, F)).reshape(-1, T, F)
```

📌 **PENTING:** `scaler.fit()` hanya dari **train**. Jangan fit ulang dari val/test.

#### Langkah 3: Train VAE-LSTM

```python
# Arsitektur
Encoder:   Input(10,5) → LSTM(64) → fc_mu(16), fc_logvar(16)
Decoder:   z(16) → fc(64) → repeat 10 → LSTM(64) → fc(5) → Output(10,5)

# Loss function
recon_loss = MSE(input, output)     # seberapa mirip hasil fotokopi
kl_loss    = KLD(μ, σ)             # regularisasi biar rapi
total_loss = recon_loss + 0.001 * kl_loss

# Training loop
for epoch in range(200):
    for batch in train_loader:
        output, mu, logvar = model(batch)
        loss = vae_loss(output, batch, mu, logvar)
        loss.backward()
        optimizer.step()
    
    # Cek loss val setiap epoch — kalau naik terus, stop lebih awal
```

**Parameter training:**

| Parameter | Nilai | Catatan |
|---|---|---|
| epochs | 200 | Bisa early stop |
| batch_size | 256 | Sesuaikan RAM |
| learning_rate | 1e-3 | Adam optimizer |
| beta (β) | 0.001 | Bobot KL loss |
| hidden_dim | 64 | Ukuran LSTM |
| latent_dim | 16 | Ukuran ringkasan |

#### Langkah 4-5: Latent + SVDD
```python
# Extract latent (pakai μ, tanpa sampling)
model.eval()
z_train = model.encode(X_train_scaled)   # ambil mu → (N, 16)

# Train SVDD
from sklearn.svm import OneClassSVM
svdd = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
svdd.fit(z_train)
```

#### Langkah 6: Threshold dari Val Set
```python
# Encode val
z_val = model.encode(X_val_scaled)
X_val_recon = model.decode(z_val)

# Hitung reconstruction error per window
recon_error = np.mean((X_val_scaled - X_val_recon) ** 2, axis=(1,2))

# Hitung SVDD distance
svdd_scores = svdd.decision_function(z_val)
svdd_dist = -svdd_scores   # makin besar = makin anomali

# Gabungkan
combined = recon_error + svdd_dist

# Threshold = persentil 95
threshold = np.percentile(combined, 95)
print(f"Threshold: {threshold:.4f}")
```

#### Langkah 7: Evaluasi
```python
# Test set (sudah ada anomali sintetik)
z_test = model.encode(X_test_scaled)
X_test_recon = model.decode(z_test)

recon_error = np.mean((X_test_scaled - X_test_recon) ** 2, axis=(1,2))
svdd_scores = svdd.decision_function(z_test)
svdd_dist = -svdd_scores
combined = recon_error + svdd_dist

y_pred = (combined > threshold).astype(int)

from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, combined):.3f}")
```

**Target hasil yang baik:**

| Metrik | Target | Keterangan |
|---|---|---|
| Accuracy | > 0.90 | Keseluruhan benar |
| Precision | > 0.85 | Yang dibilang anomali, bener anomali |
| Recall | > 0.85 | Anomali yang ke deteksi |
| F1-Score | > 0.85 | Rata-rata precision + recall |
| AUC-ROC | > 0.95 | Kemampuan bedain normal vs anomali |

Jika hasil di bawah target → cek ulang data, tuning parameter, atau tambah epoch.

#### Langkah 8: Simpan Artifacts

```python
# Struktur folder models/vae-svdd/
models/vae-svdd/
├── vae_model.pt         ← bobot VAE-LSTM (PyTorch)
├── svdd_model.pkl       ← OneClassSVM (joblib)
├── scaler.pkl           ← StandardScaler (joblib)
├── config.json          ← threshold, feature_names, dll
└── metrics.json          ← hasil evaluasi (opsional)
```

### ✅ Verifikasi Tahap 2

```bash
# Cek artifacts tersimpan
ls -lh models/vae-svdd/

# Test load model
python -c "
import torch
from modelling.vae_lstm import VAE
model = VAE()
checkpoint = torch.load('models/vae-svdd/vae_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print('Model loaded OK')
print(f'Input dim: {checkpoint[\"input_dim\"]}')
print(f'Latent dim: {checkpoint[\"latent_dim\"]}')
"

# Cek threshold
python -c "
import json
with open('models/vae-svdd/config.json') as f:
    cfg = json.load(f)
print(f'Threshold: {cfg[\"threshold\"]:.4f}')
print(f'Features: {cfg[\"feature_names\"]}')
"
```

### ❌ Common Mistakes Tahap 2

| Masalah | Akibat | Solusi |
|---|---|---|
| Scaler fit dari semua data | Bocor, skor validasi palsu | Fit hanya dari train |
| Split per record (bukan flight) | Data leakage | Split per flight_id |
| SVDD nu terlalu besar | Banyak false positive | nu=0.05 (5% data dianggap outlier) |
| Beta KLD terlalu besar | Model jadi "malas" rekonstruksi | beta=0.001 |
| Tidak monitor loss val | Overfitting | Cek loss val tiap epoch |

---

## 🥉 TAHAP 3: Inference (`serving/api.py`)

### Tujuan
Menerima data real-time tiap 30 detik, skor tiap window, kirim ke dashboard.

### Alur Inference

```
Record baru masuk (via Kafka/API)
        │
        ↓
Cari buffer pesawat ini (deque, maxlen=10)
        │
        ↓
Append record ke buffer
        │
        ↓
Buffer penuh? (isi=10)
   │         │
   TIDAK     YA
     │       │
   skip     Scale pakai scaler
     │       │
     │       ↓
     │     Encode VAE → z
     │       │
     │       ↓
     │     Hitung recon_error + svdd_dist
     │       │
     │       ↓
     │     combined > threshold?
     │    ╱        ╲
     │   YA        TIDAK
     │   │          │
     │   ↓          ↓
     │ ANOMALI    NORMAL
     │   │          │
     │   ↓          ↓
     │ Kirim ke dashboard via SSE
     │   │
     └───┘
```

```python
# Buffer per pesawat
from collections import deque, defaultdict

buffers = defaultdict(lambda: deque(maxlen=10))

def process_record(record):
    icao24 = record["icao24"]
    buf = buffers[icao24]
    
    # Ambil 5 fitur
    features = [
        record["latitude"],
        record["longitude"], 
        record["velocity"],
        record["baro_altitude"],
        record["true_track"]
    ]
    buf.append(features)
    
    # Cek apakah buffer sudah penuh
    if len(buf) < 10:
        return None  # belum cukup data
    
    # Scale
    window = np.array(buf)  # (10, 5)
    window_scaled = scaler.transform(window.reshape(1, -1)).reshape(1, 10, 5)
    
    # Model prediction
    z = model.encode(window_scaled)
    recon = model.decode(z)
    recon_error = np.mean((window_scaled - recon) ** 2)
    svdd_dist = -svdd.decision_function(z)[0]
    combined = recon_error + svdd_dist
    
    is_anomaly = combined > threshold
    
    return {
        "icao24": icao24,
        "anomaly_score": float(combined),
        "is_anomaly": bool(is_anomaly),
        "timestamp": record["timestamp"]
    }
```

### SSE Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

@app.get("/stream")
async def stream_anomalies():
    async def event_generator():
        while True:
            result = check_for_new_anomaly()  # ambil dari buffer
            if result:
                yield f"data: {json.dumps(result)}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### ✅ Verifikasi Tahap 3

```bash
# Test endpoint
curl -X POST http://localhost:8001/predict/stream \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": -6.2, "longitude": 106.8,
    "velocity": 250, "baro_altitude": 10000,
    "true_track": 90
  }'

# Response yang diharapkan
# {
#   "is_anomaly": false,
#   "anomaly_score": 0.45,
#   "attack_type": null
# }

# Test SSE endpoint
curl -N http://localhost:8001/stream
```

### ❌ Common Mistakes Tahap 3

| Masalah | Akibat | Solusi |
|---|---|---|
| Lupa scale pakai scaler yang sama | Skor kacau | Simpan scaler.pkl, load sekali |
| Buffer tidak dibersihkan | Memory leak | Deque maxlen=10 otomatis |
| Blocking request | Response lambat | Async, queue-based processing |
| Threshold dari train set | Terlalu ketat/ longgar | Threshold dari val set |

---

## 📋 Checklist Lengkap — Ceklis Sebelum Jalan

### Tahap 1: Data

- [ ] Query 14 hari terakhir dari ES
- [ ] Buang `on_ground=True`
- [ ] Filter velocity [0, 350], altitude [-500, 20000], true_track [0, 360]
- [ ] Segmentasi flight per gap > 30 menit
- [ ] Buang flight < 10 record
- [ ] Sampling ~150 flight per region
- [ ] **Split per flight_id** (70/15/15)
- [ ] Sliding window (10, stride 5)
- [ ] Inject 6 anomali ke test set
- [ ] Simpan train.npy, val.npy, test_anomaly.npy

### Tahap 2: Training

- [ ] Load data .npy
- [ ] **Scaler fit dari train SAJA**
- [ ] Train VAE-LSTM, monitor loss val
- [ ] Extract latent z dari train
- [ ] Train OneClassSVM
- [ ] Hitung threshold dari val (persentil 95)
- [ ] Evaluasi test → cek precision, recall, F1, AUC-ROC
- [ ] Simpan vae_model.pt, svdd_model.pkl, scaler.pkl, config.json

### Tahap 3: Inference

- [ ] Load artifacts sekali saat startup
- [ ] Buffer per icao24 (deque maxlen=10)
- [ ] Scale pakai scaler yang sama
- [ ] Score tiap window penuh
- [ ] Kirim hasil ke dashboard via SSE
- [ ] Cek tidak ada memory leak

---

## 🚨 Troubleshooting Cepat

| Masalah | Kemungkinan | Cek |
|---|---|---|
| Loss VAE tidak turun | Learning rate terlalu besar/kecil | Cek range loss tiap epoch |
| Semua jadi anomali | Threshold terlalu rendah | Cek distribusi combined score di val |
| Tidak ada anomali | Threshold terlalu tinggi | Cek persentil yang dipakai |
| Precision rendah | Banyak false positive | Naikkan threshold/turunkan nu SVDD |
| Recall rendah | Banyak false negative | Turunkan threshold/naikkan nu SVDD |
| AUC-ROC rendah | Data train tercemar anomali | Pastikan train 100% normal |
| Inference lambat | Batch size tidak optimal | Batch inference beberapa pesawat |

---

## 📎 Referensi

- [VAE Paper (Kingma & Welling, 2014)](https://arxiv.org/abs/1312.6114)
- [Deep SVDD (Ruff et al., 2018)](https://arxiv.org/abs/1802.03903)
- [OneClassSVM - sklearn docs](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
- [LSTM Networks (Hochreiter & Schmidhuber, 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)

---

> **Ingat:** Garbage in = garbage out. Data yang bersih dan splitting yang benar adalah 80% keberhasilan modelling. Luangkan waktu di Tahap 1, jangan terburu-buru.
