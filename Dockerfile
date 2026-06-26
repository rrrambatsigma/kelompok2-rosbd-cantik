FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch CPU-only (versi sesuai local: 2.12.1)
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY ingest/ ./ingest/
COPY preprocessing/ ./preprocessing/
COPY modelling/ ./modelling/
COPY serving/ ./serving/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY eta_scheduler.py .
COPY eta_pipeline.py .

CMD ["python", "-u", "ingest/ingester.py"]
