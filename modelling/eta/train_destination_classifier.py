import pandas as pd
import numpy as np
import os
import joblib
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from xgboost import XGBClassifier

INPUT_FILE = "data/final/eta_training_merged.parquet"
MODEL_DIR = "models"
RANDOM_STATE = 42
MAX_SAMPLES_PER_DEST = 10000

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
print("Loading dataset...")
df = pd.read_parquet(INPUT_FILE)
print("Shape:", df.shape)

# =========================
# LOAD AIRPORTS
# =========================
print("Loading airports...")
airports = pd.read_csv("data/final/airport_lookup.csv")
airport_dict = dict(zip(airports["icao"], list(zip(airports["lat"], airports["lon"]))))
print("Airports count:", len(airport_dict))

R = 6371.0

def haversine_vec(lat1, lon1, lat2, lon2):
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def bearing_vec(lat1, lon1, lat2, lon2):
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlon_r = np.radians(lon2 - lon1)
    x = np.sin(dlon_r) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon_r)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360

# =========================
# FILTER: only airborne
# =========================
print("\nFiltering data...")
df = df[df["onground"] == 0].copy()
df = df[df["altitude"] > 500].copy()

# sample 1 point per flight
# take point closest to progress_ratio = 0.7
print("\nSampling 1 point per flight at ~70% progress...")
flight_key = df["icao24"].astype(str) + "_" + df["firstseen"].astype(str)
df["flight_key"] = flight_key

def take_mid_progress(group):
    target = 0.7
    idx = (group["progress_ratio"] - target).abs().idxmin()
    return group.loc[idx]

sampled = df.groupby("flight_key").apply(take_mid_progress).reset_index(drop=True)
print("After sampling:", sampled.shape)

# =========================
# BUILD FEATURES
# =========================
print("\nBuilding features...")

X = pd.DataFrame()
X["latitude"] = sampled["latitude"].values
X["longitude"] = sampled["longitude"].values
X["altitude"] = sampled["altitude"].values
X["heading"] = sampled["heading"].values

y = sampled["arrival_airport"].values

# add distance to all airports
print("Computing airport distances...")
airport_icaos = list(airport_dict.keys())
airport_lats = np.array([airport_dict[ap][0] for ap in airport_icaos])
airport_lons = np.array([airport_dict[ap][1] for ap in airport_icaos])

dist_matrix = haversine_vec(
    X["latitude"].values[:, None],
    X["longitude"].values[:, None],
    airport_lats[None, :],
    airport_lons[None, :]
)

nearest_5_idx = np.argsort(dist_matrix, axis=1)[:, :5]
for i in range(5):
    X[f"dist_ap_{i}"] = dist_matrix[np.arange(len(dist_matrix)), nearest_5_idx[:, i]]
    X[f"bearing_ap_{i}"] = bearing_vec(
        X["latitude"].values,
        X["longitude"].values,
        airport_lats[nearest_5_idx[:, i]],
        airport_lons[nearest_5_idx[:, i]]
    )

# heading_diff = how much heading differs from bearing to nearest airport
X["heading_diff_0"] = np.abs(X["heading"] - X["bearing_ap_0"])
X["heading_diff_0"] = X["heading_diff_0"].apply(lambda x: min(x, 360 - x))

print("Feature shape:", X.shape)
print("Features:", list(X.columns))

# =========================
# BALANCED SAMPLING
# =========================
print("\nBalanced sampling per destination...")
unique_dests = np.unique(y)
print(f"Unique destinations: {len(unique_dests)}")

sampled_parts = []
for i, dest in enumerate(unique_dests):
    if i % 20 == 0:
        print(f"  Processing {i}/{len(unique_dests)}")
    mask = y == dest
    n_available = mask.sum()
    if n_available < 50:
        continue
    if dest not in airport_dict:
        continue
    n_samples = min(n_available, MAX_SAMPLES_PER_DEST)
    indices = np.where(mask)[0]
    chosen = np.random.choice(indices, n_samples, replace=False)
    part = X.iloc[chosen].copy()
    part["destination"] = dest
    sampled_parts.append(part)

X_balanced = pd.concat(sampled_parts, ignore_index=True)
y_balanced = X_balanced["destination"].values
X_balanced = X_balanced.drop(columns=["destination"])

print("Balanced shape:", X_balanced.shape)
print("Classes:", len(np.unique(y_balanced)))

# =========================
# ENCODE TARGET
# =========================
print("\nEncoding destination...")
dest_encoder = LabelEncoder()
y_encoded = dest_encoder.fit_transform(y_balanced)
print("Classes count:", len(dest_encoder.classes_))

encoder_path = os.path.join(MODEL_DIR, "destination_encoder.pkl")
joblib.dump(dest_encoder, encoder_path)
print("Saved:", encoder_path)

# =========================
# TRAIN TEST SPLIT
# =========================
print("\nTrain-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_encoded,
    test_size=0.2,
    random_state=RANDOM_STATE
)
print("Train:", X_train.shape)
print("Test :", X_test.shape)

# =========================
# TRAIN MODEL
# =========================
print("\nTraining XGBoost Classifier...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
print("\nPredicting...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
top3_acc = top_k_accuracy_score(y_test, y_proba, k=3, labels=range(len(dest_encoder.classes_)))
top5_acc = top_k_accuracy_score(y_test, y_proba, k=5, labels=range(len(dest_encoder.classes_)))

print("\n===== EVALUATION =====")
print(f"Accuracy     : {acc:.4f}")
print(f"Top-3 Acc    : {top3_acc:.4f}")
print(f"Top-5 Acc    : {top5_acc:.4f}")

feature_importance = pd.DataFrame({
    "feature": X_balanced.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# =========================
# SAVE MODEL
# =========================
model_path = os.path.join(MODEL_DIR, "destination_classifier.pkl")
joblib.dump(model, model_path)
print("\nModel saved:", model_path)

# =========================
# SAMPLE DEMO
# =========================
print("\n===== SAMPLE PREDICTIONS =====")
sample_idx = np.random.choice(len(X_test), 10, replace=False)
correct = 0
for idx in sample_idx:
    feats = X_test.iloc[idx]
    true_dest = dest_encoder.inverse_transform([y_test[idx]])[0]
    pred_proba = model.predict_proba([feats.values])[0]
    top3_idx = np.argsort(pred_proba)[-3:][::-1]
    top3 = [(dest_encoder.classes_[i], pred_proba[i]) for i in top3_idx]
    is_correct = top3[0][0] == true_dest
    correct += is_correct
    print(f"\n{'✓' if is_correct else '✗'} True: {true_dest}")
    print(f"  Pos: {feats['latitude']:.3f}, {feats['longitude']:.3f}, Alt: {feats['altitude']:.0f}, Hdg: {feats['heading']:.0f}")
    for dest, prob in top3:
        print(f"  {dest}: {prob:.3f}")

print(f"\nSample accuracy: {correct}/{len(sample_idx)}")

print("\nDONE.")
