import pandas as pd

# Supaya pandas nggak motong output
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

# Load parquet
df = pd.read_parquet("data/historical/test_100_rows.parquet")

# =========================
# INFO DASAR
# =========================
print("===== SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
print(df.columns)

print("\n===== DTYPES =====")
print(df.dtypes)

# =========================
# LIHAT 1 ROW LENGKAP
# =========================
print("\n===== ROW PERTAMA =====")
print(df.iloc[0])

# =========================
# CEK TRACK
# =========================
track = df.iloc[0]["track"]

print("\n===== TRACK INFO =====")
print("Type track:", type(track))

# kalau track list / array
try:
    print("Jumlah titik track:", len(track))
except:
    print("Track tidak punya len()")

# tampilkan 3 titik pertama
print("\n===== 3 TITIK PERTAMA TRACK =====")
try:
    for i, point in enumerate(track[:3]):
        print(f"Point {i+1}:")
        print(point)
        print()
except:
    print(track)

# =========================
# CEK NILAI NULL
# =========================
print("\n===== NULL COUNT =====")
print(df.isnull().sum())