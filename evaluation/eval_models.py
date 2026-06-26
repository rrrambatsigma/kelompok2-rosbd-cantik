import pandas as pd
import numpy as np
import joblib
import os
import sys

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

DATA_FILE = os.path.join(BASE, "data/final/eta_training_merged.parquet")
AIRPORT_FILE = os.path.join(BASE, "data/final/airport_lookup.csv")
MODEL_DIR = os.path.join(BASE, "models")

np.random.seed(42)
R = 6371.0

def haversine_vec(lat1, lon1, lat2, lon2):
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat/2)**2 + np.cos(lat1_r)*np.cos(lat2_r)*np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

print("Loading models...")
clf = joblib.load(f"{MODEL_DIR}/destination_classifier.pkl")
dest_encoder = joblib.load(f"{MODEL_DIR}/destination_encoder.pkl")
eta_model = joblib.load(f"{MODEL_DIR}/eta_xgboost_balanced.pkl")
route_encoder = joblib.load(f"{MODEL_DIR}/route_encoder.pkl")

print("Loading data...")
df = pd.read_parquet(DATA_FILE)
print(f"Dataset: {df.shape}")

airports = pd.read_csv(AIRPORT_FILE)
airport_dict = dict(zip(airports["icao"], list(zip(airports["lat"], airports["lon"]))))
airport_icaos = list(airport_dict.keys())
airport_lats = np.array([airport_dict[ap][0] for ap in airport_icaos])
airport_lons = np.array([airport_dict[ap][1] for ap in airport_icaos])

# ============================================================
# 1. DESTINATION CLASSIFIER EVALUATION
# ============================================================
print("\n" + "=" * 55)
print("  EVALUATING DESTINATION CLASSIFIER...")
print("=" * 55)

clf_df = df[df["onground"] == 0].copy()
clf_df = clf_df[clf_df["altitude"] > 500].copy()

flight_key = clf_df["icao24"].astype(str) + "_" + clf_df["firstseen"].astype(str)
clf_df["flight_key"] = flight_key

def take_mid_progress(group):
    target = 0.7
    idx = (group["progress_ratio"] - target).abs().idxmin()
    return group.loc[idx]

sampled = clf_df.groupby("flight_key").apply(take_mid_progress).reset_index(drop=True)
print(f"  Sampled {len(sampled)} flights")

X_clf = pd.DataFrame()
X_clf["latitude"] = sampled["latitude"].values.astype(float)
X_clf["longitude"] = sampled["longitude"].values.astype(float)
X_clf["altitude"] = sampled["altitude"].values.astype(float)
X_clf["heading"] = sampled["heading"].values.astype(float)
y_clf_raw = sampled["arrival_airport"].values

known = set(dest_encoder.classes_)
mask = np.isin(y_clf_raw, list(known))
X_clf = X_clf[mask].reset_index(drop=True)
y_clf_raw = y_clf_raw[mask]
print(f"  Known classes: {len(X_clf)} samples")

print("  Computing features...")
dist_matrix = haversine_vec(
    X_clf["latitude"].values[:, None],
    X_clf["longitude"].values[:, None],
    airport_lats[None, :],
    airport_lons[None, :]
)
nearest_5_idx = np.argsort(dist_matrix, axis=1)[:, :5]
for i in range(5):
    X_clf[f"dist_ap_{i}"] = dist_matrix[np.arange(len(dist_matrix)), nearest_5_idx[:, i]]
    lat1 = X_clf["latitude"].values
    lon1 = X_clf["longitude"].values
    lat2 = airport_lats[nearest_5_idx[:, i]]
    lon2 = airport_lons[nearest_5_idx[:, i]]
    lat1_r = np.radians(lat1); lat2_r = np.radians(lat2)
    dlon_r = np.radians(lon2 - lon1)
    x = np.sin(dlon_r) * np.cos(lat2_r)
    y = np.cos(lat1_r)*np.sin(lat2_r) - np.sin(lat1_r)*np.cos(lat2_r)*np.cos(dlon_r)
    X_clf[f"bearing_ap_{i}"] = (np.degrees(np.arctan2(x, y)) + 360) % 360

hd = np.abs(X_clf["heading"] - X_clf["bearing_ap_0"])
X_clf["heading_diff_0"] = np.array([min(h, 360-h) for h in hd])

print("  Splitting data...")
indices = np.arange(len(X_clf))
np.random.shuffle(indices)
split = int(len(indices) * 0.8)
train_idx, test_idx = indices[:split], indices[split:]
X_test_clf = X_clf.iloc[test_idx].reset_index(drop=True)
y_test_clf = y_clf_raw[test_idx]

print(f"  Test samples: {len(X_test_clf)}")

print("  Predicting...")
y_pred_clf = clf.predict(X_test_clf.values)
y_proba_clf = clf.predict_proba(X_test_clf.values)

print("\n  Computing metrics...")
n_test = len(y_test_clf)
y_test_enc = dest_encoder.transform(y_test_clf)

acc_val = accuracy_score(y_test_enc, y_pred_clf)
top3 = accuracy_score(y_test_enc, y_pred_clf)  # placeholder
topk_correct = 0
for i in range(len(y_test_enc)):
    top_k = np.argsort(y_proba_clf[i])[-3:]
    if y_test_enc[i] in top_k:
        topk_correct += 1
top3 = topk_correct / len(y_test_enc)
topk_correct5 = 0
for i in range(len(y_test_enc)):
    top_k = np.argsort(y_proba_clf[i])[-5:]
    if y_test_enc[i] in top_k:
        topk_correct5 += 1
top5 = topk_correct5 / len(y_test_enc)

macro_p = precision_score(y_test_enc, y_pred_clf, average="macro", zero_division=0)
macro_r = recall_score(y_test_enc, y_pred_clf, average="macro", zero_division=0)
macro_f1 = f1_score(y_test_enc, y_pred_clf, average="macro", zero_division=0)
weighted_p = precision_score(y_test_enc, y_pred_clf, average="weighted", zero_division=0)
weighted_r = recall_score(y_test_enc, y_pred_clf, average="weighted", zero_division=0)
weighted_f1 = f1_score(y_test_enc, y_pred_clf, average="weighted", zero_division=0)

correct_mask = y_pred_clf == y_test_enc
conf_correct = np.mean(y_proba_clf[np.arange(n_test), y_pred_clf][correct_mask]) if correct_mask.sum() > 0 else 0
conf_wrong = np.mean(y_proba_clf[np.arange(n_test), y_pred_clf][~correct_mask]) if (~correct_mask).sum() > 0 else 0

# ============================================================
# 2. ETA REGRESSOR EVALUATION
# ============================================================
print("\n" + "=" * 55)
print("  EVALUATING ETA REGRESSOR...")
print("=" * 55)

print("  Preparing data...")
eta_df = df[["latitude","longitude","altitude","heading",
             "elapsed_time","progress_ratio","route","remaining_time"]].dropna()

known_routes = set(route_encoder.classes_)
eta_df = eta_df[eta_df["route"].isin(known_routes)]

if len(eta_df) > 30000:
    eta_df = eta_df.sample(n=30000, random_state=42)

float_cols = ["latitude","longitude","altitude","heading",
              "elapsed_time","progress_ratio","remaining_time"]
for c in float_cols:
    if c in eta_df.columns:
        eta_df[c] = eta_df[c].astype("float32")

eta_df["route"] = route_encoder.transform(eta_df["route"])
feat_cols = ["latitude","longitude","altitude","heading",
             "elapsed_time","progress_ratio","route"]

print(f"  Test samples: {len(eta_df)}")

print("  Predicting...")
X_eta = eta_df[feat_cols]
y_eta_true = eta_df["remaining_time"].values
y_eta_pred = eta_model.predict(X_eta)

mae_val = mean_absolute_error(y_eta_true, y_eta_pred)
rmse_val = np.sqrt(mean_squared_error(y_eta_true, y_eta_pred))
r2_val = r2_score(y_eta_true, y_eta_pred)
errors = np.abs(y_eta_true - y_eta_pred)

# ============================================================
# 3. PRINT SUMMARY
# ============================================================
print("\n\n")
print("=" * 55)
print("           MODEL EVALUATION SUMMARY")
print("=" * 55)

print(f"""
-----------------------------------------------
  1. DESTINATION CLASSIFIER (XGBoost)
-----------------------------------------------
  Test samples         : {n_test:,}
  Classes              : {len(dest_encoder.classes_)}

  Accuracy             : {acc_val:.4f}  ({acc_val*100:.2f}%)
  Top-3 Accuracy       : {top3:.4f}  ({top3*100:.2f}%)
  Top-5 Accuracy       : {top5:.4f}  ({top5*100:.2f}%)

  Macro Avg Precision  : {macro_p:.4f}
  Macro Avg Recall     : {macro_r:.4f}
  Macro Avg F1         : {macro_f1:.4f}

  Weighted Avg Precision : {weighted_p:.4f}
  Weighted Avg Recall    : {weighted_r:.4f}
  Weighted Avg F1        : {weighted_f1:.4f}

  Confidence (correct) : {conf_correct:.4f}
  Confidence (wrong)   : {conf_wrong:.4f}

-----------------------------------------------
  2. ETA REGRESSOR (XGBoost)
-----------------------------------------------
  Test samples         : {len(eta_df):,}

  MAE                  : {mae_val:.2f} detik  ({mae_val/60:.2f} menit)
  RMSE                 : {rmse_val:.2f} detik  ({rmse_val/60:.2f} menit)
  R2 Score             : {r2_val:.4f}

  Error Distribution:
    < 1 menit          : {(errors < 60).mean()*100:.1f}%
    < 5 menit          : {(errors < 300).mean()*100:.1f}%
    < 10 menit         : {(errors < 600).mean()*100:.1f}%
    < 15 menit         : {(errors < 900).mean()*100:.1f}%
    < 30 menit         : {(errors < 1800).mean()*100:.1f}%
""")
print("=" * 55)
