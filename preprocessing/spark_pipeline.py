import os
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.ml.feature import StandardScaler as SparkStandardScaler
from pyspark.ml.feature import VectorAssembler

FEATURE_COLUMNS = [
    "longitude", "latitude", "velocity",
    "geo_altitude", "true_track", "vertical_rate"
]


def create_spark_session(app_name="VAE-SVDD"):
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.debug.maxToStringFields", "100") \
        .getOrCreate()


def fetch_from_elasticsearch(
    es_host: str = "elasticsearch:9200",
    index: str = "flights",
    size: int = 10000,
    scroll: str = "10m"
) -> pd.DataFrame:
    es = Elasticsearch(f"http://{es_host}")

    result = es.search(
        index=index,
        scroll=scroll,
        size=size,
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"exists": {"field": "latitude"}},
                        {"exists": {"field": "longitude"}},
                        {"exists": {"field": "velocity"}},
                        {"exists": {"field": "true_track"}},
                    ]
                }
            },
            "sort": [{"timestamp": {"order": "asc"}}],
            "_source": [
                "icao24", "callsign", "timestamp",
                "latitude", "longitude", "velocity",
                "geo_altitude", "true_track", "vertical_rate",
                "origin_country", "on_ground"
            ]
        }
    )

    all_docs = []
    scroll_id = result.get("_scroll_id")
    hits = result["hits"]["hits"]
    all_docs.extend([h["_source"] for h in hits])

    while len(hits) > 0:
        result = es.scroll(scroll_id=scroll_id, scroll=scroll)
        scroll_id = result.get("_scroll_id")
        hits = result["hits"]["hits"]
        all_docs.extend([h["_source"] for h in hits])

    es.clear_scroll(scroll_id=scroll_id)

    return pd.DataFrame(all_docs)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=FEATURE_COLUMNS, how="any").copy()

    df = df[df["velocity"].between(0, 350)]
    df = df[df["geo_altitude"].between(-500, 20000)]
    df = df[df["true_track"].between(0, 360)]
    df = df[df["vertical_rate"].between(-50, 50)]

    df = df.drop_duplicates(subset=["icao24", "timestamp"])

    return df.reset_index(drop=True)


def create_spark_dataframe(spark: SparkSession, pdf: pd.DataFrame) -> DataFrame:
    schema = StructType([
        StructField("icao24", StringType(), True),
        StructField("callsign", StringType(), True),
        StructField("timestamp", FloatType(), True),
        StructField("latitude", FloatType(), True),
        StructField("longitude", FloatType(), True),
        StructField("velocity", FloatType(), True),
        StructField("geo_altitude", FloatType(), True),
        StructField("true_track", FloatType(), True),
        StructField("vertical_rate", FloatType(), True),
    ])

    return spark.createDataFrame(pdf, schema=schema)


def normalize_with_spark(spark_df: DataFrame, feature_cols=None) -> DataFrame:
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="feature_vector")
    assembled = assembler.transform(spark_df)

    scaler = SparkStandardScaler(
        inputCol="feature_vector",
        outputCol="scaled_features",
        withStd=True,
        withMean=True
    )
    scaler_model = scaler.fit(assembled)
    scaled = scaler_model.transform(assembled)

    return scaled, scaler_model
