FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY ingest/ ./ingest/
COPY preprocessing/ ./preprocessing/
COPY modelling/ ./modelling/
COPY serving/ ./serving/
COPY models/ ./models/

CMD ["python", "-u", "ingestion/ingester.py"]
