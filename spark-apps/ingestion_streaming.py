import requests
import json
import time
from kafka import KafkaProducer
import os

KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = "raw_flight"

producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

OPENSKY_URL = "https://opensky-network.org/api/states/all"

def fetch_and_produce():
    while True:
        try:
            response = requests.get(OPENSKY_URL)
            if response.status_code == 200:
                data = response.json()
                states = data.get('states', [])
                for state in states:
                    record = {
                        "icao24": state[0],
                        "callsign": state[1],
                        "origin_country": state[2],
                        "time_position": state[3],
                        "last_contact": state[4],
                        "longitude": state[5],
                        "latitude": state[6],
                        "baro_altitude": state[7],
                        "on_ground": state[8],
                        "velocity": state[9],
                        "true_track": state[10],
                        "vertical_rate": state[11],
                        "geo_altitude": state[13],
                        "timestamp": time.time()
                    }
                    producer.send(TOPIC, value=record)
                print(f"Produced {len(states)} records")
            else:
                print(f"OpenSky API error: {response.status_code}")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(10)

if __name__ == "__main__":
    fetch_and_produce()