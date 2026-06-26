"""
detector_es.py — Ambil data real-time dari ES teman langsung
Alternatif dari detector.py (yang pake Kafka)

Alur:
  1. Polling ES teman tiap 5 detik untuk data baru
  2. Kirim setiap record ke serving API /predict/stream
  3. Serving API buffer → VAE → deteksi anomali → ES → Grafana
"""
import os
import sys
import time
import json
import requests
from datetime import datetime
from elasticsearch import Elasticsearch

REMOTE_ES = os.getenv("REMOTE_ES", "http://100.99.130.69:9200")
REMOTE_INDEX = os.getenv("REMOTE_INDEX", "flights")
LOCAL_SERVING = os.getenv("SERVING_URL", "http://localhost:8001")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))

es = None
last_seen = 0
stats = {"polled": 0, "sent": 0, "errors": 0, "anomalies": 0}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def connect_es():
    global es
    try:
        es = Elasticsearch(REMOTE_ES, request_timeout=10)
        info = es.info()
        log(f"Connected to ES teman: {info['name']} ({info['version']['number']})")
        log(f"Index '{REMOTE_INDEX}': {es.count(index=REMOTE_INDEX)['count']:,} docs")
        return True
    except Exception as e:
        log(f"ES connection failed: {e}")
        return False


def get_latest_timestamp():
    """Ambil ingested_at terbaru dari ES teman buat starting point."""
    try:
        result = es.search(
            index=REMOTE_INDEX,
            body={
                "size": 1,
                "sort": [{"ingested_at": {"order": "desc"}}],
                "_source": ["ingested_at"]
            }
        )
        hits = result["hits"]["hits"]
        if hits:
            return hits[0]["_source"]["ingested_at"]
    except Exception as e:
        log(f"Failed to latest ingested_at: {e}")
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"


def poll_new_data():
    """Query data baru dari ES teman (ingested_at > last_seen)."""
    global last_seen
    try:
        result = es.search(
            index=REMOTE_INDEX,
            body={
                "query": {
                    "range": {
                        "ingested_at": {"gt": last_seen}
                    }
                },
                "sort": [{"ingested_at": {"order": "asc"}}],
                "size": BATCH_SIZE,
                "_source": [
                    "icao24", "callsign", "latitude", "longitude",
                    "velocity", "baro_altitude", "true_track", "timestamp", "ingested_at"
                ]
            }
        )
        return [h["_source"] for h in result["hits"]["hits"]]
    except Exception as e:
        log(f"Poll error: {e}")
        return []


def send_to_serving(flight):
    """Kirim 1 record ke serving API."""
    try:
        resp = requests.post(
            f"{LOCAL_SERVING}/predict/stream",
            json={
                "icao24": flight.get("icao24"),
                "callsign": flight.get("callsign"),
                "latitude": flight.get("latitude", 0),
                "longitude": flight.get("longitude", 0),
                "velocity": flight.get("velocity", 0),
                "baro_altitude": flight.get("baro_altitude", 0),
                "true_track": flight.get("true_track", 0),
            },
            timeout=3,
        )
        if resp.status_code == 200:
            result = resp.json()
            if result.get("is_anomaly"):
                stats["anomalies"] += 1
                log(f"ANOMALI: {flight.get('icao24')} - {result.get('attack_type')} (recon={result.get('recon_error'):.3f})")
            return True
        else:
            log(f"API error {resp.status_code}: {resp.text[:100]}")
            stats["errors"] += 1
            return False
    except requests.ConnectionError:
        log(f"API connection error - serving API down?")
        stats["errors"] += 1
        return False
    except Exception as e:
        log(f"Send error: {e}")
        stats["errors"] += 1
        return False


def print_stats():
    log(f"Stats: {stats['polled']} polled, {stats['sent']} sent, "
        f"{stats['anomalies']} anomalies, {stats['errors']} errors")


def main():
    log("=" * 55)
    log("DETECTOR-ES: Anomaly Detection via ES Teman")
    log(f"  ES teman:  {REMOTE_ES}/{REMOTE_INDEX}")
    log(f"  Serving:   {LOCAL_SERVING}")
    log(f"  Interval:  {POLL_INTERVAL}s")
    log("=" * 55)

    if not connect_es():
        log("Gagal connect ke ES teman. Cek Tailscale / IP.")
        return

    global last_seen
    last_seen = get_latest_timestamp()
    log(f"Starting from ingested_at: {last_seen}")
    log(f"Menunggu data real-time dari ES teman...")
    log("=" * 55)

    last_stats_time = time.time()

    while True:
        records = poll_new_data()
        stats["polled"] += len(records)

        for flight in records:
            # Update last_seen ke ingested_at record ini
            ingested = flight.get("ingested_at")
            if ingested and ingested > last_seen:
                last_seen = ingested

            # Skip data di darat kalau ada flag on_ground
            if flight.get("on_ground"):
                continue

            # Kirim ke serving API
            if send_to_serving(flight):
                stats["sent"] += 1

        # Print stats tiap 30 detik
        if time.time() - last_stats_time > 30:
            print_stats()
            last_stats_time = time.time()
            # Cek status serving API
            try:
                health = requests.get(f"{LOCAL_SERVING}/health", timeout=3)
                if health.status_code != 200:
                    log(f"Serving API health check failed: {health.status_code}")
            except:
                log(f"Serving API tidak bisa diakses!")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
