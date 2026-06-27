from kafka import KafkaConsumer, KafkaAdminClient
from kafka.client_async import KafkaClient
import json

BROKER = "100.99.130.69:9093"
TOPIC = "flights"

print("=== CEK TOPIC VIA ADMIN ===")
try:
    admin = KafkaAdminClient(bootstrap_servers=BROKER)
    topics = admin.list_topics()
    print(f"Topics: {topics}")
except Exception as e:
    print(f"Admin error: {e}")

print()
print("=== CEK TOPIC VIA CLIENT ===")
try:
    client = KafkaClient(bootstrap_servers=BROKER)
    client.poll(timeout=10)
    topics = client.cluster.topics()
    print(f"Topics from cluster: {topics}")
    client.close()
except Exception as e:
    print(f"Client error: {e}")

print()
print(f"=== CONSUMER TEST (topic={TOPIC}, timeout=60s) ===")
print(f"Menunggu data dari Meiva...")

c = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BROKER,
    auto_offset_reset="latest",
    consumer_timeout_ms=60000,
    request_timeout_ms=65000,
)

count = 0
for msg in c:
    count += 1
    print(f"\nDATA MASUK! Message #{count}")
    val = msg.value
    print(f"Type: {type(val).__name__}")

    if isinstance(val, dict):
        first_key = list(val.keys())[0]
        first_val = val[first_key]
        print(f"Sample '{first_key}': type={type(first_val).__name__}, val={str(first_val)[:100]}")
        print(f"All keys: {list(val.keys())}")
        print(f"Full message:")
        print(json.dumps(val, indent=2)[:1000])
    else:
        print(f"Raw content: {str(val)[:500]}")
    break

if count == 0:
    print("Tidak ada data dalam 60 detik.")
    print()
    print("Kemungkinan penyebab:")
    print("1. Meiva sedang tidak menjalankan ingesternya")
    print("2. Topic mungkin beda (bukan 'flights')")
    print("3. Kafka butuh autentikasi (SASL/SSL)")

c.close()
print("\nSelesai.")
