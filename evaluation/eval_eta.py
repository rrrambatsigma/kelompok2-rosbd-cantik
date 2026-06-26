import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

INPUT_FILE = "data/final/eta_training_merged.parquet"
MODEL_DIR = "models"
RANDOM_STATE = 42
MAX_EVAL_SAMPLES = 50000
os.makedirs("eval_output", exist_ok=True)

# =========================
# LOAD MODEL & ENCODER
# =========================
print("Loading model & encoder...")
model = joblib.load(f"{MODEL_DIR}/eta_xgboost_balanced.pkl")
route_encoder = joblib.load(f"{MODEL_DIR}/route_encoder.pkl")
known_routes = set(route_encoder.classes_)
print(f"Known routes: {len(known_routes)}")

# =========================
# LOAD DATA
# =========================
print("Loading dataset...")
df = pd.read_parquet(INPUT_FILE)
print("Shape:", df.shape)

# =========================
# FEATURE SELECTION
# =========================
features = [
    "latitude", "longitude", "altitude", "heading",
    "elapsed_time", "progress_ratio", "route"
]
target = "remaining_time"

df = df[features + [target]].dropna()
print("After dropna:", df.shape)

# Filter known routes
df = df[df["route"].isin(known_routes)]
print(f"After filtering known routes: {df.shape}")

# =========================
# MEMORY OPTIMIZATION
# =========================
float_cols = ["latitude", "longitude", "altitude", "heading",
              "elapsed_time", "progress_ratio", "remaining_time"]
for col in float_cols:
    if col in df.columns:
        df[col] = df[col].astype("float32")

# =========================
# SAMPLE FOR EVAL
# =========================
if len(df) > MAX_EVAL_SAMPLES:
    df = df.sample(n=MAX_EVAL_SAMPLES, random_state=RANDOM_STATE)
    print(f"Sampled to {MAX_EVAL_SAMPLES} for eval")

# =========================
# ENCODE ROUTE (keep column name "route" to match training)
# =========================
df["route"] = route_encoder.transform(df["route"])
feature_cols = ["latitude", "longitude", "altitude", "heading",
                "elapsed_time", "progress_ratio", "route"]

# =========================
# TRAIN TEST SPLIT
# =========================
print("\nTrain-test split...")
X = df[feature_cols]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Test set: {len(X_test)} samples")

# =========================
# PREDICT
# =========================
print("\nPredicting...")
y_pred = model.predict(X_test)

# =========================
# OVERALL METRICS
# =========================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
errors = np.abs(y_test - y_pred)

print("\n\n========================================")
print("   OVERALL METRICS")
print("========================================")
print(f"MAE (seconds)          : {mae:.2f}")
print(f"RMSE (seconds)         : {rmse:.2f}")
print(f"MAE (minutes)          : {mae/60:.2f}")
print(f"RMSE (minutes)         : {rmse/60:.2f}")
print(f"R² Score               : {r2:.4f}")

# =========================
# ERROR DISTRIBUTION
# =========================
print("\n\n========================================")
print("   ERROR DISTRIBUTION")
print("========================================")
bins = [0, 60, 120, 300, 600, 900, 1800, 3600, 999999]
bin_labels = ["<1m", "1-2m", "2-5m", "5-10m", "10-15m", "15-30m", "30-60m", ">60m"]
for i in range(len(bins)-1):
    count = ((errors >= bins[i]) & (errors < bins[i+1])).sum()
    pct = count / len(errors) * 100
    print(f"  {bin_labels[i]:>8}: {count:>6} ({pct:>5.1f}%)")

# cumulative
print()
for threshold_min, label in [(1, "1 menit"), (5, "5 menit"), (10, "10 menit"),
                              (15, "15 menit"), (30, "30 menit")]:
    pct = (errors < threshold_min * 60).mean() * 100
    print(f"  Error < {label:>10}: {pct:.1f}%")

# =========================
# PER-ROUTE METRICS
# =========================
print("\n\n========================================")
print("   PER-ROUTE METRICS (top 20 by MAE)")
print("========================================")
test_df = X_test.copy()
test_df["true"] = y_test.values
test_df["pred"] = y_pred
test_df["error"] = np.abs(y_test.values - y_pred)
test_df["route_name"] = route_encoder.inverse_transform(X_test["route"].astype(int))

route_stats = test_df.groupby("route").agg(
    mae_sec=("error", "mean"),
    rmse_sec=("pred", lambda x: np.sqrt(mean_squared_error(
        test_df.loc[x.index, "true"], x))),
    samples=("error", "count"),
    true_mean=("true", "mean")
).reset_index()

route_stats["mae_min"] = route_stats["mae_sec"] / 60
route_stats["rmse_min"] = route_stats["rmse_sec"] / 60
route_stats = route_stats.sort_values("mae_sec", ascending=False)

print(f"{'Route':<20} {'MAE(min)':>10} {'RMSE(min)':>10} {'Samples':>8} {'AvgDur(min)':>12}")
print("-" * 60)
for _, r in route_stats.head(20).iterrows():
    print(f"{r['route']:<20} {r['mae_min']:>10.2f} {r['rmse_min']:>10.2f} {r['samples']:>8} {r['true_mean']/60:>12.1f}")

route_stats.to_csv("eval_output/eta_per_route_metrics.csv", index=False)

# =========================
# BEST & WORST ROUTES
# =========================
print("\n\n===== WORST 10 ROUTES (highest MAE) =====")
worst = route_stats[route_stats["samples"] >= 50].head(10)
for _, r in worst.iterrows():
    print(f"  {r['route']:<20} MAE: {r['mae_min']:>6.2f}m   Samples: {r['samples']:>5}")

print("\n===== BEST 10 ROUTES (lowest MAE) =====")
best = route_stats[route_stats["samples"] >= 50].tail(10).sort_values("mae_sec")
for _, r in best.iterrows():
    print(f"  {r['route']:<20} MAE: {r['mae_min']:>6.2f}m   Samples: {r['samples']:>5}")

# =========================
# ERROR BY PROGRESS
# =========================
print("\n\n========================================")
print("   ERROR BY FLIGHT PROGRESS")
print("========================================")
test_df["progress_bin"] = pd.cut(test_df["progress_ratio"],
                                 bins=[0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
                                 labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-90%", "90-100%"])
progress_stats = test_df.groupby("progress_bin", observed=True).agg(
    mae_sec=("error", "mean"),
    samples=("error", "count")
).reset_index()
progress_stats["mae_min"] = progress_stats["mae_sec"] / 60
print(f"{'Progress':<12} {'MAE(min)':>10} {'Samples':>8}")
print("-" * 32)
for _, r in progress_stats.iterrows():
    print(f"{r['progress_bin']:<12} {r['mae_min']:>10.2f} {r['samples']:>8}")

# =========================
# ACTUAL VS PREDICTED PLOT
# =========================
print("\n\nGenerating plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Actual vs Predicted scatter
ax = axes[0]
sample_plot = test_df.sample(min(5000, len(test_df)))
ax.scatter(sample_plot["true"]/60, sample_plot["pred"]/60, alpha=0.3, s=1)
ax.plot([0, test_df["true"].max()/60], [0, test_df["true"].max()/60], "r--", linewidth=1)
ax.set_xlabel("Actual (minutes)")
ax.set_ylabel("Predicted (minutes)")
ax.set_title("Actual vs Predicted ETA")
ax.set_aspect("equal")

# 2. Error histogram
ax = axes[1]
ax.hist(errors/60, bins=100, alpha=0.7, color="steelblue")
ax.set_xlabel("Absolute Error (minutes)")
ax.set_ylabel("Frequency")
ax.set_title(f"Error Distribution (MAE={mae/60:.1f}m)")
ax.axvline(mae/60, color="red", linestyle="--", label=f"Mean={mae/60:.1f}m")
ax.legend()

# 3. Error by progress
ax = axes[2]
ax.bar(progress_stats["progress_bin"].astype(str), progress_stats["mae_min"], color="steelblue")
ax.set_xlabel("Flight Progress")
ax.set_ylabel("MAE (minutes)")
ax.set_title("Error by Flight Progress")
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("eval_output/eta_eval_plots.png", dpi=150)
print("Saved: eval_output/eta_eval_plots.png")

# =========================
# SUMMARY
# =========================
print("\n\n========================================")
print("   SUMMARY")
print("========================================")
print(f"Test samples       : {len(X_test)}")
print(f"MAE                : {mae:.2f}s ({mae/60:.2f}m)")
print(f"RMSE               : {rmse:.2f}s ({rmse/60:.2f}m)")
print(f"R2 Score           : {r2:.4f}")
print(f"Error < 5 menit    : {(errors < 300).mean()*100:.1f}%")
print(f"Error < 15 menit   : {(errors < 900).mean()*100:.1f}%")
print(f"Output files:")
print(f"  eval_output/eta_eval_plots.png")
print(f"  eval_output/eta_per_route_metrics.csv")

# Save overall metrics JSON
import json
overall = {
    "mae_seconds": round(float(mae), 2),
    "mae_minutes": round(float(mae/60), 2),
    "rmse_seconds": round(float(rmse), 2),
    "rmse_minutes": round(float(rmse/60), 2),
    "r2_score": round(float(r2), 4),
    "error_lt_1min_pct": round(float((errors < 60).mean() * 100), 1),
    "error_lt_5min_pct": round(float((errors < 300).mean() * 100), 1),
    "error_lt_10min_pct": round(float((errors < 600).mean() * 100), 1),
    "error_lt_15min_pct": round(float((errors < 900).mean() * 100), 1),
    "error_lt_30min_pct": round(float((errors < 1800).mean() * 100), 1),
    "test_samples": len(X_test)
}
with open("eval_output/regressor_overall.json", "w") as f:
    json.dump(overall, f, indent=2)
print(f"  eval_output/regressor_overall.json")

print("\nDONE.")
