# ============================================================
# RETRAIN MODEL VAE-LSTM + SVDD dari ES Meiva
# ============================================================
# Jalankan dari folder project:
#   .\run_retrain.ps1
# ============================================================

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "RETRAIN VAE-LSTM + SVDD - Data dari ES Meiva" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# ── Cek Prasyarat ──
$ErrorActionPreference = "Stop"

if (-not (Test-Path "venv/Scripts/python.exe")) {
    Write-Host "[ERROR] Virtual env tidak ditemukan. Jalankan dulu:" -ForegroundColor Red
    Write-Host "  python -m venv venv" -ForegroundColor Yellow
    Write-Host "  .\venv\Scripts\Activate" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

$DATA_DIR = "data/checkpoint"
$MODEL_DIR = "models/vae-svdd"
$LOG_DIR = "logs"

# Backup model lama
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
if (Test-Path $MODEL_DIR/vae_model.pt) {
    $backup_dir = "models/vae-svdd-backup-$timestamp"
    New-Item -ItemType Directory -Path $backup_dir -Force | Out-Null
    Copy-Item "$MODEL_DIR/*" $backup_dir -Force
    Write-Host "[BACKUP] Model lama disimpan ke $backup_dir" -ForegroundColor Yellow
}

# Aktifkan virtual env
& ".\venv\Scripts\Activate.ps1"

# ── Konfigurasi (SETELAH activate) ──
$env:ES_HOST = "100.99.130.69:9200"
$env:ES_INDEX = "flights"

Write-Host ""
Write-Host "ES Meiva: http://$env:ES_HOST/$env:ES_INDEX" -ForegroundColor Green
Write-Host ""

# ── Step 1: Fetch + Generate Dataset ──
Write-Host "════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "STEP 1/2: Fetch + Generate Dataset dari ES Meiva" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

python preprocessing/dump_data.py 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] dump_data.py gagal!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[OK] Dataset siap!" -ForegroundColor Green
Write-Host ""

# ── Cek hasil dataset ──
Write-Host "Cek file dataset:" -ForegroundColor Yellow
python -c "
import numpy as np
import os
data_dir = 'data/checkpoint'
files = ['train_data.npy', 'val_data.npy', 'test_data.npy', 'test_labels.npy']
for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        arr = np.load(path)
        print(f'  {f}: {arr.shape}', end='')
        if 'label' in f:
            print(f' (anomaly: {arr.mean()*100:.1f}%)')
        else:
            print(f' (range: {arr.min():.2f} - {arr.max():.2f})')
    else:
        print(f'  {f}: NOT FOUND')
"

# ── Step 2: Training Model ──
Write-Host ""
Write-Host "════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "STEP 2/2: Training VAE-LSTM + OneClassSVM" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

python -m modelling.anomaly.train `
    --vae-epochs 200 `
    --latent-dim 16 `
    --svdd-nu 0.05 `
    --batch-size 256 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] train.py gagal!" -ForegroundColor Red
    exit 1
}

# ── Cek hasil training ──
Write-Host ""
Write-Host "════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "HASIL TRAINING" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════" -ForegroundColor Cyan

python -c "
import json
import os

# Cek metrics
metrics_path = 'models/vae-svdd/metrics.json'
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        m = json.load(f)
    print(f'Best method: {m.get(\"best_method\", \"?\")}')
    print(f'Accuracy:  {m.get(\"accuracy\", 0)*100:.1f}%')
    print(f'Precision: {m.get(\"precision\", 0)*100:.1f}%')
    print(f'Recall:    {m.get(\"recall\", 0)*100:.1f}%')
    print(f'F1-Score:  {m.get(\"f1\", 0)*100:.1f}%')
    print(f'AUC-ROC:   {m.get(\"auc\", 0)*100:.1f}%')

# Cek config
config_path = 'models/vae-svdd/config.json'
if os.path.exists(config_path):
    with open(config_path) as f:
        c = json.load(f)
    print(f'Threshold: {c.get(\"best_threshold\", \"?\")}')

# Cek file model
for f in ['vae_model.pt', 'svdd_model.pkl', 'scaler.pkl']:
    path = os.path.join('models/vae-svdd', f)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    print(f'  {f}: {size/1024:.1f} KB' if size > 0 else f'  {f}: NOT FOUND')
"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "SELESAI!" -ForegroundColor Green
Write-Host "Model baru tersimpan di: models/vae-svdd/" -ForegroundColor Green
Write-Host ""
Write-Host "Cara restart API dengan model baru:" -ForegroundColor Yellow
Write-Host "  `$env:ELASTICSEARCH_HOST='localhost:9200'" -ForegroundColor Yellow
Write-Host "  uvicorn serving.api:app --host 0.0.0.0 --port 8001" -ForegroundColor Yellow
Write-Host ""
Write-Host "Cek hasil training:" -ForegroundColor Yellow
Write-Host "  curl http://localhost:8001/health" -ForegroundColor Yellow
Write-Host "  curl http://localhost:8001/model/info" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan
