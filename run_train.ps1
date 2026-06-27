# ============================================================
# RUN TRAINING — VAE-LSTM + SVDD
# ============================================================
# Jalankan dari folder project (VENV SUDAH AKTIF):
#   .\run_train.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$MODEL_DIR = "models/vae-svdd"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "TRAINING VAE-LSTM + OneClassSVM" -ForegroundColor Cyan
Write-Host "Dataset: data/checkpoint/ (existing)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# ── Cek dataset ──
Write-Host ""
Write-Host "Cek dataset..." -ForegroundColor Yellow
python -c "
import numpy as np, os
path = 'data/checkpoint'
files = ['train_data.npy','val_data.npy','test_data.npy','test_labels.npy']
for f in files:
    p = os.path.join(path, f)
    if os.path.exists(p):
        d = np.load(p)
        print(f'  {f}: {d.shape}')
    else:
        print(f'  {f}: NOT FOUND - STOP')
        exit(1)
print('Dataset OK')
"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Dataset tidak lengkap!" -ForegroundColor Red
    Write-Host "Jalankan dulu: python preprocessing/dump_data.py" -ForegroundColor Yellow
    exit 1
}

# ── Training ──
Write-Host ""
Write-Host "════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Mulai training (15-25 menit)..." -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

python -m modelling.anomaly.train --epochs 200 --latent-dim 16 --nu 0.05
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Training gagal!" -ForegroundColor Red
    exit 1
}

# ── Hasil ──
Write-Host ""
Write-Host "════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "HASIL TRAINING" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════" -ForegroundColor Cyan

python -c "
import json
m = json.load(open('models/vae-svdd/metrics.json'))
c = json.load(open('models/vae-svdd/config.json'))
print()
print(f'Metode terbaik: {m[\"best_method\"]}')
print(f'Threshold:      {c[\"best_threshold\"]}')
print(f'Accuracy:       {m[\"accuracy\"]*100:.2f}%')
print(f'Precision:      {m[\"precision\"]*100:.2f}%')
print(f'Recall:         {m[\"recall\"]*100:.2f}%')
print(f'F1-Score:       {m[\"f1\"]*100:.2f}%')
print(f'AUC-ROC:        {m[\"auc\"]*100:.2f}%')
print()
print('Model baru di: models/vae-svdd/')
print('Model lama di: models/vae-svdd-backup-*')
"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "SELESAI!" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Restart API dengan model baru:" -ForegroundColor Yellow
Write-Host '  $env:ELASTICSEARCH_HOST="localhost:9200"' -ForegroundColor Yellow
Write-Host "  uvicorn serving.api:app --host 0.0.0.0 --port 8001" -ForegroundColor Yellow
