import os
import sys
import time
import json
import requests
from datetime import datetime

REMOTE_ES = os.getenv("REMOTE_ES", "http://100.99.130.69:9200")
REMOTE_INDEX = "flights"
SERVING_URL = os.getenv("SERVING_URL", "http://localhost:8001")
BATCH_SIZE = 2000

stats = {"fetched": 0, "sent": 0, "errors": 0, "anomalies": 0}
t_start = time.time()
buffer_status_url = f"{SERVING_URL}/buffer/status"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

def fetch_from_es():
    log(f"Fetching {BATCH_SIZE} records from {REMOTE_ES}/{REMOTE_INDEX}...")
    try:
        resp = requests.get(
            f"{REMOTE_ES}/{REMOTE_INDEX}/_search",
            json={
                "size": BATCH_SIZE,
                "query": {"match_all": {}},
                "sort": [{"timestamp": {"order": "desc"}}],
                "_source": [
                    "icao24", "callsign", "latitude", "longitude",
                    "velocity", "baro_altitude", "true_track", "timestamp"
                ]
            },
            timeout=30
        )
        if resp.status_code != 200:
            log(f"ES fetch failed: {resp.status_code} {resp.text[:200]}")
            return []

        hits = resp.json()["hits"]["hits"]
        records = [h["_source"] for h in hits]
        log(f"Fetched {len(records)} records from ES Meiva")

        sorted_records = sorted(records, key=lambda r: (r.get("icao24",""), r.get("timestamp",0)))
        return sorted_records

    except Exception as e:
        log(f"ES fetch error: {e}")
        return []

def send_to_model(record):
    try:
        payload = {
            "icao24": record.get("icao24"),
            "callsign": record.get("callsign", ""),
            "latitude": float(record.get("latitude", 0)),
            "longitude": float(record.get("longitude", 0)),
            "velocity": float(record.get("velocity", 0)),
            "baro_altitude": float(record.get("baro_altitude", 0)),
            "true_track": float(record.get("true_track", 0)),
        }
        resp = requests.post(
            f"{SERVING_URL}/predict/stream",
            json=payload,
            timeout=5
        )
        if resp.status_code == 200:
            result = resp.json()
            if result.get("is_anomaly"):
                stats["anomalies"] += 1
                icao = record.get("icao24", "?")
                attack = result.get("attack_type", "?")
                recon = result.get("recon_error", 0)
                log(f"ANOMALI! icao24={icao} attack={attack} recon={recon:.4f}")
            return True
        else:
            log(f"API error {resp.status_code}: {resp.text[:100]}")
            stats["errors"] += 1
            return False
    except requests.exceptions.ConnectionError:
        stats["errors"] += 1
        return False
    except Exception as e:
        log(f"Send error: {e}")
        stats["errors"] += 1
        return False

def main():
    log("=" * 55)
    log("FETCH ES MEIVA - VAE-LSTM Anomaly Detection")
    log(f"  ES Meiva: {REMOTE_ES}/{REMOTE_INDEX}")
    log(f"  Serving:  {SERVING_URL}")
    log(f"  Batch:    {BATCH_SIZE} records")
    log("=" * 55)

    records = fetch_from_es()
    if not records:
        log("No records fetched. Exiting.")
        return

    stats["fetched"] = len(records)
    log(f"Sending {len(records)} records to model (sorted by icao24+timestamp)...")

    icao_groups = {}
    for r in records:
        icao = r.get("icao24", "unknown")
        if icao not in icao_groups:
            icao_groups[icao] = []
        icao_groups[icao].append(r)

    log(f"Unique flights: {len(icao_groups)}")
    log(f"Min records per flight: {min(len(v) for v in icao_groups.values())}")
    log(f"Max records per flight: {max(len(v) for v in icao_groups.values())}")

    for icao, flight_records in icao_groups.items():
        flight_records.sort(key=lambda r: r.get("timestamp", 0))

    last_report = time.time()
    total_sent = 0

    for icao, flight_records in icao_groups.items():
        for record in flight_records:
            send_to_model(record)
            total_sent += 1
            stats["sent"] += 1

            if time.time() - last_report > 5:
                elapsed = time.time() - t_start
                rate = total_sent / elapsed if elapsed > 0 else 0
                log(f"Progress: {total_sent}/{stats['fetched']} sent "
                     f"({rate:.1f}/s), {stats['anomalies']} anomalies, "
                     f"{stats['errors']} errors")
                last_report = time.time()

    elapsed = time.time() - t_start
    try:
        buf = requests.get(buffer_status_url, timeout=3).json()
        active = buf.get("active_buffers", "?")
        log(f"Buffer status after send: {active} active buffers")
    except:
        pass

    log("=" * 55)
    log(f"DONE in {elapsed:.1f}s")
    log(f"  Fetched:  {stats['fetched']}")
    log(f"  Sent:     {stats['sent']}")
    log(f"  Anomalies:{stats['anomalies']}")
    log(f"  Errors:   {stats['errors']}")
    log("=" * 55)
    log("Cek hasil di ES: curl http://localhost:9200/anomaly-stream/_count")
    log("Atau Grafana: http://localhost:3000")

if __name__ == "__main__":
    main()
