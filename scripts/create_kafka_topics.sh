#!/bin/bash
echo "Waiting for Kafka..."
sleep 10
kafka-topics --create --topic raw_flight --bootstrap-server kafka:9092 --partitions 3 --replication-factor 1 || true
kafka-topics --create --topic processed_flight --bootstrap-server kafka:9092 --partitions 3 --replication-factor 1 || true
kafka-topics --create --topic anomaly_notifications --bootstrap-server kafka:9092 --partitions 1 --replication-factor 1 || true
echo "Topics created."