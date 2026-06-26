import pandas as pd
import numpy as np
import joblib
import math
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    top_k_accuracy_score
)
import matplotlib.pyplot as plt

INPUT_FILE = "data/final/eta_training_merged.parquet"
MODEL_DIR = "models"
RANDOM_STATE = 42
os.makedirs("eval_output", exist_ok=True)

# =========================
# LOAD MODEL & ENCODER
# =========================
print("Loading model & encoder...")
clf = joblib.load(f"{MODEL_DIR}/destination_classifier.pkl")
dest_encoder = joblib.load(f"{MODEL_DIR}/destination_encoder.pkl")
print(f"Model classes: {len(dest_encoder.classes_)}")

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
airport_icaos = list(airport_dict.keys())
airport_lats = np.array([airport_dict[ap][0] for ap in airport_icaos])
airport_lons = np.array([airport_dict[ap][1] for ap in airport_icaos])

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
# SAMPLE 1 POINT PER FLIGHT
# =========================
print("\nSampling 1 point per flight at ~70% progress...")
df = df[df["onground"] == 0].copy()
df = df[df["altitude"] > 500].copy()

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
X["latitude"] = sampled["latitude"].values.astype(float)
X["longitude"] = sampled["longitude"].values.astype(float)
X["altitude"] = sampled["altitude"].values.astype(float)
X["heading"] = sampled["heading"].values.astype(float)
y_raw = sampled["arrival_airport"].values

known_classes = set(dest_encoder.classes_)
mask = np.isin(y_raw, list(known_classes))
X = X[mask].reset_index(drop=True)
y_raw = y_raw[mask]
print(f"After filtering known classes: {len(X)} samples, {len(np.unique(y_raw))} classes")

print("Computing airport distances...")
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

X["heading_diff_0"] = np.abs(X["heading"] - X["bearing_ap_0"])
X["heading_diff_0"] = X["heading_diff_0"].apply(lambda x: min(x, 360 - x))

print("Feature shape:", X.shape)

# =========================
# ENCODE & SPLIT
# =========================
y_encoded = dest_encoder.transform(y_raw)

print("\nTrain-test split...")
indices = np.arange(len(X))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE)
X_train = X.iloc[train_idx].reset_index(drop=True)
X_test = X.iloc[test_idx].reset_index(drop=True)
y_train = y_encoded[train_idx]
y_test = y_encoded[test_idx]
y_train_raw = y_raw[train_idx]
y_test_raw = y_raw[test_idx]
print("Test set size:", len(X_test))

# =========================
# PREDICT
# =========================
print("\nPredicting...")
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# =========================
# CLASSIFICATION REPORT
# =========================
print("\n\n========================================")
print("   CLASSIFICATION REPORT")
print("========================================")
target_names = dest_encoder.classes_
report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
print(report)

with open("eval_output/classification_report.txt", "w") as f:
    f.write("CLASSIFICATION REPORT\n\n")
    f.write(report)

# =========================
# ACCURACY
# =========================
acc = accuracy_score(y_test, y_pred)
top3 = top_k_accuracy_score(y_test, y_proba, k=3, labels=range(len(target_names)))
top5 = top_k_accuracy_score(y_test, y_proba, k=5, labels=range(len(target_names)))

print(f"\nAccuracy     : {acc:.4f}")
print(f"Top-3 Acc    : {top3:.4f}")
print(f"Top-5 Acc    : {top5:.4f}")

# =========================
# PER-CLASS METRICS SUMMARY
# =========================
print("\n\n========================================")
print("   PER-CLASS METRICS (sorted by f1)")
print("========================================")
report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
per_class = []
for cls in target_names:
    if cls in report_dict:
        per_class.append({
            "airport": cls,
            "precision": round(report_dict[cls]["precision"], 4),
            "recall": round(report_dict[cls]["recall"], 4),
            "f1": round(report_dict[cls]["f1-score"], 4),
            "support": int(report_dict[cls]["support"])
        })
per_class_df = pd.DataFrame(per_class)
per_class_df = per_class_df.sort_values("f1", ascending=False).reset_index(drop=True)
pd.set_option('display.max_rows', None)
print(per_class_df.to_string(index=False))
pd.reset_option('display.max_rows')

per_class_df.to_csv("eval_output/per_class_metrics.csv", index=False)

# =========================
# WORST 10
# =========================
print("\n\n===== WORST 10 CLASSES (by f1) =====")
worst = per_class_df[per_class_df["support"] > 5].tail(10)
print(worst.to_string(index=False))

# =========================
# CONFUSION MATRIX ERRORS
# =========================
print("\n\n===== TOP MISCLASSIFICATIONS =====")
cm = confusion_matrix(y_test, y_pred)
np.fill_diagonal(cm, 0)
error_pairs = []
for i in range(len(target_names)):
    for j in range(len(target_names)):
        if cm[i][j] > 0:
            error_pairs.append({
                "true": target_names[i],
                "predicted": target_names[j],
                "count": cm[i][j]
            })
error_df = pd.DataFrame(error_pairs).sort_values("count", ascending=False)
print(error_df.head(20).to_string(index=False))
error_df.to_csv("eval_output/confusion_errors.csv", index=False)

# =========================
# CONFIDENCE ANALYSIS
# =========================
print("\n\n===== CONFIDENCE ANALYSIS =====")
correct_mask = y_pred == y_test
conf_correct = y_proba[np.arange(len(y_proba)), y_pred][correct_mask]
conf_wrong = y_proba[np.arange(len(y_proba)), y_pred][~correct_mask]
print(f"Mean confidence (correct): {conf_correct.mean():.4f}")
print(f"Mean confidence (wrong):   {conf_wrong.mean():.4f}")
print(f"Min confidence (correct):  {conf_correct.min():.4f}")
print(f"Max confidence (wrong):    {conf_wrong.max():.4f}")

# =========================
# CONFUSION MATRIX PLOT (top 20)
# =========================
print("\n\nGenerating confusion matrix plot...")
top20_classes = [c["airport"] for c in per_class[:20]]
mask_top20 = np.isin(y_test_raw, top20_classes)
y_test_top20_raw = y_test_raw[mask_top20]
y_pred_top20_raw = dest_encoder.inverse_transform(y_pred[mask_top20])

# Clip both to top20 only
both_in_top20 = np.isin(y_test_top20_raw, top20_classes) & np.isin(y_pred_top20_raw, top20_classes)
y_test_top20_raw = y_test_top20_raw[both_in_top20]
y_pred_top20_raw = y_pred_top20_raw[both_in_top20]

from sklearn.preprocessing import LabelEncoder
le_top20 = LabelEncoder()
le_top20.fit(top20_classes)
y_test_top20_enc = le_top20.transform(y_test_top20_raw)
y_pred_top20_enc = le_top20.transform(y_pred_top20_raw)

cm_top20 = confusion_matrix(y_test_top20_enc, y_pred_top20_enc, labels=range(len(top20_classes)))
disp = ConfusionMatrixDisplay(cm_top20, display_labels=le_top20.classes_)
fig, ax = plt.subplots(figsize=(16, 14))
disp.plot(ax=ax, xticks_rotation=90, cmap="Blues")
plt.tight_layout()
plt.savefig("eval_output/confusion_matrix_top20.png", dpi=150)
print("Saved: eval_output/confusion_matrix_top20.png")

# =========================
# SUMMARY
# =========================
print("\n\n========================================")
print("   SUMMARY")
print("========================================")
print(f"Total test samples: {len(X_test)}")
print(f"Accuracy          : {acc:.4f} ({acc*100:.2f}%)")
print(f"Top-3 Accuracy    : {top3:.4f} ({top3*100:.2f}%)")
print(f"Top-5 Accuracy    : {top5:.4f} ({top5*100:.2f}%)")
print(f"Classes           : {len(target_names)}")
print(f"Correct mean conf : {conf_correct.mean():.4f}")
print(f"Wrong mean conf   : {conf_wrong.mean():.4f}")
print(f"\nOutput files:")
print(f"  eval_output/classification_report.txt")
print(f"  eval_output/per_class_metrics.csv")
print(f"  eval_output/confusion_errors.csv")
print(f"  eval_output/confusion_matrix_top20.png")

# Save overall metrics JSON
import json
report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
overall = {
    "accuracy": round(acc, 4),
    "top3_accuracy": round(top3, 4),
    "top5_accuracy": round(top5, 4),
    "macro_avg_precision": round(report_dict["macro avg"]["precision"], 4),
    "macro_avg_recall": round(report_dict["macro avg"]["recall"], 4),
    "macro_avg_f1": round(report_dict["macro avg"]["f1-score"], 4),
    "weighted_avg_precision": round(report_dict["weighted avg"]["precision"], 4),
    "weighted_avg_recall": round(report_dict["weighted avg"]["recall"], 4),
    "weighted_avg_f1": round(report_dict["weighted avg"]["f1-score"], 4),
    "test_samples": int(report_dict["macro avg"]["support"]),
    "n_classes": len(target_names),
    "confidence_correct": round(float(conf_correct.mean()), 4),
    "confidence_wrong": round(float(conf_wrong.mean()), 4)
}
with open("eval_output/classifier_overall.json", "w") as f:
    json.dump(overall, f, indent=2)
print(f"  eval_output/classifier_overall.json")

print("\nDONE.")
