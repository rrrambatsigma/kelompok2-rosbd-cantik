from kafka import KafkaConsumer
import json, os
import numpy as np
from sklearn.svm import OneClassSVM

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

# Model SVDD sederhana dengan data dummy
model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
X_train = np.random.rand(100, 4)  # latih dengan fitur dummy
model.fit(X_train)

consumer = KafkaConsumer(
    "flights",
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest'
)

print("Anomaly detector started...")
for msg in consumer:
    flight = msg.value
    # Ekstrak fitur: longitude, latitude, velocity, geo_altitude
    features = np.array([[
        flight.get('longitude', 0),
        flight.get('latitude', 0),
        flight.get('velocity', 0),
        flight.get('geo_altitude', 0)
    ]])
    prediction = model.predict(features)
    if prediction[0] == -1:
        print(f"⚠️ Anomaly detected: {flight.get('icao24')} - {flight.get('callsign')}")
    else:
        print(f"✅ Normal: {flight.get('icao24')}")