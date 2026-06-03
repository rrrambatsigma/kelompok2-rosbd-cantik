from prefect import flow, task
import subprocess
import time

@task
def run_spark_ingestion():
    # Jalankan Spark streaming job (bisa dijalankan di background)
    subprocess.Popen(["spark-submit", "/spark-apps/ingestion_streaming.py"])

@task
def run_spark_preprocessing():
    subprocess.Popen(["spark-submit", "--packages", "org.apache.sedona:sedona-spark-shaded-3.5_2.12:1.5.1,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0", "/spark-apps/preprocessing_with_sedona.py"])

@task
def monitor_kafka():
    # Cek health Kafka, dll
    print("Monitoring Kafka...")
    time.sleep(5)

@flow(name="OpenSky Pipeline")
def main_flow():
    run_spark_ingestion()
    run_spark_preprocessing()
    monitor_kafka()

if __name__ == "__main__":
    main_flow()