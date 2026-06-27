# FIX

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# =========================
# CONFIG
# =========================
INPUT_FILE = "data/final/eta_training_merged.parquet"
MODEL_DIR = "models"

RANDOM_STATE = 42
MAX_SAMPLES_PER_ROUTE = 5000

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
print("Loading dataset...")
df = pd.read_parquet(INPUT_FILE)

print("Original shape:")
print(df.shape)

print("\nUnique routes:")
print(df["route"].nunique())

# =========================
# FEATURE SELECTION
# =========================
features = [
    "latitude",
    "longitude",
    "altitude",
    "heading",
    "elapsed_time",
    "progress_ratio",
    "route"
]

target = "remaining_time"

df = df[features + [target]].dropna()

print("\nAfter feature selection:")
print(df.shape)

# =========================
# MEMORY OPTIMIZATION
# =========================
print("\nOptimizing memory...")

float_cols = [
    "latitude",
    "longitude",
    "altitude",
    "heading",
    "elapsed_time",
    "progress_ratio",
    "remaining_time"
]

for col in float_cols:
    df[col] = df[col].astype("float32")

print("Done.")

# =========================
# BALANCED SAMPLING
# =========================
print("\nBalanced sampling per route...")

sampled_parts = []

unique_routes = df["route"].unique()
total_routes = len(unique_routes)

for i, route_name in enumerate(unique_routes):

    if i % 25 == 0:
        print(f"Processing route {i}/{total_routes}")

    route_data = df[df["route"] == route_name]

    n_samples = min(
        len(route_data),
        MAX_SAMPLES_PER_ROUTE
    )

    sampled_route = route_data.sample(
        n=n_samples,
        random_state=RANDOM_STATE
    )

    sampled_parts.append(sampled_route)

df = pd.concat(
    sampled_parts,
    ignore_index=True
)

print("\nAfter balanced sampling:")
print(df.shape)

print("\nColumns after sampling:")
print(df.columns)

route_counts = df["route"].value_counts()

print("\nRoute stats:")
print("Min samples :", route_counts.min())
print("Max samples :", route_counts.max())
print("Mean samples:", route_counts.mean())

# =========================
# ENCODE ROUTE
# =========================
print("\nEncoding route...")

route_encoder = LabelEncoder()

df["route"] = route_encoder.fit_transform(
    df["route"]
)

encoder_path = os.path.join(
    MODEL_DIR,
    "route_encoder.pkl"
)

joblib.dump(
    route_encoder,
    encoder_path
)

print("Route encoder saved:")
print(encoder_path)

# =========================
# TRAIN TEST SPLIT
# =========================
X = df[features]
y = df[target]

print("\nTrain-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# =========================
# TRAIN MODEL
# =========================
print("\nTraining XGBoost...")

model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

# =========================
# EVALUATION
# =========================
print("\nPredicting...")

pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)

print("\n===== EVALUATION =====")
print(f"MAE  : {mae:.2f} seconds")
print(f"RMSE : {rmse:.2f} seconds")
print(f"MAE in minutes: {mae/60:.2f}")

# =========================
# FEATURE IMPORTANCE
# =========================
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# =========================
# SAVE MODEL
# =========================
model_path = os.path.join(
    MODEL_DIR,
    "eta_xgboost_balanced.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\nModel saved:")
print(model_path)