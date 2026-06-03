# Dockerfile
@'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "eta_api:app", "--host", "0.0.0.0", "--port", "8001"]
'@ | Out-File -FilePath backend\Dockerfile -Encoding ascii

# requirements.txt
@'
fastapi
uvicorn
xgboost
scikit-learn
elasticsearch
kafka-python
numpy
pandas
'@ | Out-File -FilePath backend\requirements.txt -Encoding ascii

# eta_api.py
@'
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "ETA API running"}
'@ | Out-File -FilePath backend\eta_api.py -Encoding ascii

# optimasi_api.py
@'
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Optimasi API running"}
'@ | Out-File -FilePath backend\optimasi_api.py -Encoding ascii

# anomali_api.py
@'
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Anomali API running"}
'@ | Out-File -FilePath backend\anomali_api.py -Encoding ascii

# notification.py
@'
print("Notification service started")
import time
while True:
    time.sleep(10)
'@ | Out-File -FilePath backend\notification.py -Encoding ascii