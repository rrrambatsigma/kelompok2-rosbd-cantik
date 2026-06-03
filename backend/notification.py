import json
from kafka import KafkaConsumer
import os
import time

KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = "anomaly_notifications"

def send_notification(anomaly_data):
    # Bisa kirim email, telegram, atau print ke console
    print(f"[NOTIFICATION] Anomaly detected: {anomaly_data}")

def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset='earliest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    print(f"Listening to {TOPIC}...")
    for msg in consumer:
        anomaly_data = msg.value
        send_notification(anomaly_data)

if __name__ == "__main__":
    time.sleep(10)  # Tunggu Kafka siap
    main()