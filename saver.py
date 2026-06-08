import os
import time
import json
import requests
from datetime import datetime
from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
from kafka.errors import NoBrokersAvailable

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = "flights"
GROUP_ID = "flight-saver-group"

ES_HOST = os.getenv("ELASTICSEARCH_HOST", "elasticsearch:9200")
es = Elasticsearch(f"http://{ES_HOST}")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def preprocess(flight: dict):
    """Membersihkan dan menambah field data sebelum disimpan."""
    # Skip jika tidak ada callsign atau posisi
    if not flight.get("callsign"):
        return None
    if flight.get("longitude") is None or flight.get("latitude") is None:
        return None

    flight["region"] = "Europe"

    # Konversi kecepatan dari m/s ke km/jam
    if flight.get("velocity") is not None:
        flight["velocity_kmh"] = round(flight["velocity"] * 3.6, 2)
    else:
        flight["velocity_kmh"] = None

    # Filter altitude tidak wajar (misal > 20 km)
    if flight.get("geo_altitude") and flight["geo_altitude"] > 20000:
        flight["geo_altitude"] = None

    # Timestamp proses
    flight["processed_at"] = datetime.utcnow().isoformat() + "Z"
    return flight

def send_telegram(flight: dict):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    # Bangun pesan dari semua field yang ada
    message_lines = ["✈️ *Flight Data*"]
    for key, value in flight.items():
        # Lewati field yang nilainya None (opsional)
        if value is not None:
            message_lines.append(f"`{key}`: {value}")
    message = "\n".join(message_lines)
    
    if len(message) > 4000:
        message = message[:4000] + "\n... (truncated)"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=5)
        print(f"[TELEGRAM] Notifikasi terkirim untuk {flight.get('callsign')}")
    except Exception as e:
        print(f"[TELEGRAM] Gagal kirim: {e}")

def connect_kafka():
    print(f"[KAFKA] Menghubungkan consumer ke {KAFKA_BOOTSTRAP}...")
    while True:
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id=GROUP_ID,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=True,
            )
            print("[KAFKA] Consumer berhasil terhubung")
            return consumer
        except NoBrokersAvailable:
            print("[KAFKA] Broker belum siap, tunggu 5 detik...")
            time.sleep(5)
        except Exception as e:
            print(f"[KAFKA] Error: {e}, tunggu 5 detik...")
            time.sleep(5)

def main():
    try:
        es.info()
        print("[ES] Terhubung ke Elasticsearch")
    except Exception as e:
        print(f"[ES] Gagal konek: {e}")
        exit(1)

    consumer = connect_kafka()
    print(f"[MAIN] Consumer berjalan, mendengarkan topic '{TOPIC}'")

    for msg in consumer:
        flight = msg.value
        clean = preprocess(flight)
        if clean is None:
            print("[SKIP] Data tidak valid, lewati")
            continue

        try:
            es.index(index="flights", document=clean)
            print(f"[ES] Tersimpan flight {clean.get('callsign')}")
        except Exception as e:
            print(f"[ES] Gagal simpan: {e}")

        send_telegram(clean)

if __name__ == "__main__":
    main()