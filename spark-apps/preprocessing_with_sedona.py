from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, when, lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType
from sedona.spark import SedonaContext
from sedona.core.geom.envelope import Envelope
from sedona.sql.types import GeometryType
from sedona.sql import functions as F

# Konfigurasi Sedona
spark = SparkSession.builder \
    .appName("SedonaPreprocessing") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator") \
    .config("spark.sql.extensions", "org.apache.sedona.sql.SedonaSqlExtensions") \
    .getOrCreate()

sedona = SedonaContext.create(spark)

# Baca dari Kafka
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "raw_flight") \
    .option("startingOffsets", "latest") \
    .load()

# Schema data dari OpenSky
schema = StructType([
    StructField("icao24", StringType()),
    StructField("callsign", StringType()),
    StructField("origin_country", StringType()),
    StructField("time_position", LongType()),
    StructField("last_contact", LongType()),
    StructField("longitude", DoubleType()),
    StructField("latitude", DoubleType()),
    StructField("baro_altitude", DoubleType()),
    StructField("on_ground", BooleanType()),
    StructField("velocity", DoubleType()),
    StructField("true_track", DoubleType()),
    StructField("vertical_rate", DoubleType()),
    StructField("geo_altitude", DoubleType()),
    StructField("timestamp", DoubleType())
])

# Parse JSON
parsed_df = kafka_df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

# Filter data valid (longitude, latitude tidak null)
filtered_df = parsed_df.filter(col("longitude").isNotNull() & col("latitude").isNotNull())

# Contoh geospasial: buat point geometry
geo_df = filtered_df.withColumn("geometry", F.ST_Point(col("longitude"), col("latitude")))

# Contoh bounding box: Eropa
bbox = Envelope( -10.0, 30.0, 35.0, 70.0 )  # (minX, maxX, minY, maxY)
geo_df = geo_df.withColumn("in_europe", F.ST_Within(geo_df.geometry, F.ST_GeomFromWKT("POLYGON((-10 35, 30 35, 30 70, -10 70, -10 35))")))

# Tambahkan kolom timestamp processing
from pyspark.sql.functions import current_timestamp
final_df = geo_df.withColumn("processing_time", current_timestamp())

# Write ke Elasticsearch (sink streaming)
es_write = final_df.writeStream \
    .outputMode("append") \
    .format("org.elasticsearch.spark.sql") \
    .option("es.nodes", "elasticsearch") \
    .option("es.port", "9200") \
    .option("es.resource", "opensky_flights") \
    .option("checkpointLocation", "/opt/bitnami/spark/data/checkpoint") \
    .start()

es_write.awaitTermination()