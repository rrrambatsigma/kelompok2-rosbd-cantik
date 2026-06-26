$env:KAFKA_BOOTSTRAP_SERVERS="100.99.130.69:9092"
$env:SERVING_URL="http://localhost:8001"
$env:ELASTICSEARCH_HOST="localhost:9200"
python serving/detector.py
