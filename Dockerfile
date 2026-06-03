FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .

# Tambah -u supaya output tidak di-buffer
CMD ["python", "-u", "ingester.py"]