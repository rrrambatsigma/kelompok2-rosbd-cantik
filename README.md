# Proyek Kelompok 2 ROSBD — ADS-B Anomaly Detection dengan VAE-SVDD

Sistem deteksi anomali data penerbangan ADS-B dari **OpenSky Network** menggunakan **VAE-SVDD** (Variational Autoencoder + Support Vector Data Description) dengan pipeline **Apache Spark** dan visualisasi **Grafana**.

---

## Arsitektur

```
OpenSky API → ingester → Kafka → serving/detector → vae-serving (VAE-SVDD API) → Grafana
                      ↓                          ↕
                 Elasticsearch ←──── anomaly-results, anomaly-stream
```

## 6 Jenis Anomali yang Dideteksi

| No | Attack | Feature | Metode Deteksi |
|---|---|---|---|
| 1 | Constant Position Deviation | lat/lon | Reconstruction error tinggi di posisi |
| 2 | Random Position Deviation | lat/lon | Noise di latent space |
| 3 | Velocity Drift | velocity | SVDD outlier + drift temporal |
| 4 | DoS / Message Deletion | all | Gap sequence + high recon error |
| 5 | Flight Replacement / Merge | lat,lon,vel,alt | Shift tiba-tiba di latent space |
| 6 | Heading Manipulation | true_track | Inkonsistensi track vs posisi |

---

## Struktur Direktori

```
.
├── ingest/                       ← Data ingestion pipeline
│   ├── ingester.py               (fetch OpenSky → Kafka + ES)
│   ├── saver.py                  (Kafka → ES + Telegram)
│   └── optimizer.py              (FastAPI route optimizer)
├── preprocessing/                ← Data preprocessing
│   ├── spark_pipeline.py         (PySpark + Elasticsearch pipeline)
│   └── attack_generator.py       (6 synthetic attack types)
├── modelling/                    ← Model training & core
│   ├── vae_svdd.py               (VAE + SVDD: PyTorch + sklearn)
│   └── train.py                  (Training orchestrator)
├── serving/                      ← Serving & detection
│   ├── api.py                    (FastAPI inference server)
│   └── detector.py               (Kafka consumer → serving API)
├── models/vae-svdd/              (Model artifacts: vae.pt, svdd.joblib, scaler.joblib)
├── grafana-provisioning/         (Auto-provision dashboard + datasource)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Cara Menjalankan — Langkah demi Langkah

### Prasyarat

- Docker & Docker Compose terinstal
- File `credentials.json` berisi client ID & secret OpenSky Network

### 1. Jalankan Infrastruktur

```bash
docker-compose up -d zookeeper kafka elasticsearch kibana grafana
```

Tunggu ~30 detik sampai Elasticsearch dan Kafka siap.

Verifikasi:
```bash
curl http://localhost:9200   # Elasticsearch
curl http://localhost:5601   # Kibana
```

### 2. Jalankan Data Ingestion

```bash
docker-compose up -d ingester saver optimizer
```

- **ingester**: fetch data dari OpenSky API → Kafka topic `flights` + simpan ke ES index `flights`
- **saver**: consumer Kafka → simpan ke ES + kirim notifikasi Telegram
- **optimizer**: endpoint `/optimize` di port `8000`

Cek data masuk:
```bash
curl "http://localhost:9200/flights/_count"
```

### 3. Training Model VAE-SVDD

Build image & jalankan training:

```bash
docker-compose build vae-serving
docker-compose run --rm vae-serving python -m modelling.train \
    --max-samples 50000 \
    --vae-epochs 200 \
    --latent-dim 4 \
    --svdd-nu 0.05 \
    --inject-anomalies
```

**Penjelasan parameter:**

| Parameter | Default | Fungsi |
|---|---|---|
| `--max-samples` | 50000 | Jumlah data yang diambil dari ES |
| `--vae-epochs` | 200 | Epoch training VAE |
| `--latent-dim` | 4 | Dimensi latent space |
| `--svdd-nu` | 0.05 | Expected anomaly ratio untuk SVDD |
| `--inject-anomalies` | True | Suntik 6 jenis anomali ke test set untuk evaluasi |

Training akan menampilkan:
- Loss VAE per 20 epoch
- Jumlah support vectors SVDD
- Threshold anomaly
- **Confusion matrix**: TP, FP, FN, TN + Precision, Recall, F1-Score

Model akan disimpan ke `models/vae-svdd/`:
```
models/vae-svdd/
├── vae.pt              (Bobot VAE PyTorch)
├── svdd.joblib          (OneClassSVM)
├── scaler.joblib        (StandardScaler)
└── config.json          (Konfigurasi)
```

### 4. Jalankan Serving API

```bash
docker-compose up -d vae-serving
```

Serving API berjalan di `http://localhost:8001`.

Verifikasi:
```bash
curl http://localhost:8001/health
curl http://localhost:8001/features
```

Test prediksi:
```bash
curl -X POST http://localhost:8001/predict/stream \
    -H "Content-Type: application/json" \
    -d '{
        "longitude": 106.8, "latitude": -6.2,
        "velocity": 250.0, "geo_altitude": 10000.0,
        "true_track": 90.0, "vertical_rate": 0.0
    }'
```

### 5. Jalankan Detektor Anomali Real-Time

```bash
docker-compose up -d anomaly
```

**detector** akan:
1. Konsumsi dari Kafka topic `flights`
2. Kirim tiap data point ke serving API (`/predict/stream`)
3. Simpan hasil ke ES index `anomaly-stream`
4. Log statistik setiap 30 detik

Cek hasil:
```bash
curl "http://localhost:9200/anomaly-stream/_count?q=is_anomaly:true"
```

### 6. Visualisasi di Grafana

Buka **http://localhost:3000** (login: `admin / admin`).

Dashboard **"ADS-B Anomaly Detection - VAE-SVDD"** sudah auto-provisioning dengan panel:

| Panel | Deskripsi |
|---|---|
| Anomaly Stream Count | Total anomali real-time |
| Anomaly Rate (%) | Gauge persentase anomali |
| Anomaly Score Timeline | Combined score, recon error, SVDD distance |
| Anomaly by Attack Type | Pie chart distribusi tipe anomali |
| Attack Type Over Time | Stacked timeline tiap attack |
| Anomaly Locations Map | Peta geografis lokasi anomali |
| Dominant Feature Distribution | Bar chart fitur paling berkontribusi |
| Recent Anomalies Table | Tabel 50 anomali terbaru |

### 7. Evaluasi Model (Opsional)

Lihat metrik model di ES:
```bash
curl "http://localhost:9200/model-performance/_search?pretty&q=model:vae-svdd&sort=timestamp:desc&size=1"
```

---

## Endpoint API

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/health` | Status model & service |
| `GET` | `/features` | Daftar fitur yang digunakan |
| `GET` | `/model/info` | Info konfigurasi model |
| `POST` | `/predict` | Batch prediction |
| `POST` | `/predict/stream` | Single prediction + simpan ke ES |

---

## Troubleshooting

**Problem:** Elasticsearch tidak bisa start
```bash
# Cek logs
docker-compose logs elasticsearch
# Pastikan vm.max_map_count >= 262144 (Linux host)
sudo sysctl -w vm.max_map_count=262144
```

**Problem:** Kafka connection refused
```bash
# Tunggu zookeeper siap dulu
docker-compose logs zookeeper
```

**Problem:** Serving API mengembalikan 503
```bash
# Model belum di-train. Jalankan training dulu:
docker-compose run --rm vae-serving python -m modelling.train
```

**Problem:** Tidak ada data di Elasticsearch
```bash
# Cek ingester logs
docker-compose logs ingester
# Pastikan credentials.json valid
```

---

## Catatan

- Data disimpan di ES index `flights`, hasil deteksi di `anomaly-stream`
- Model disimpan di host path `./models/vae-svdd/` (mounted volume)
- Training menggunakan PyTorch CPU (bisa diubah dengan flag `--gpu`)
- Untuk production, ganti `python:3.10-slim` dengan image yang punya CUDA support
