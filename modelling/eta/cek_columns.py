import pandas as pd

file = "data/historical/eta_training_1724544000.parquet"

df = pd.read_parquet(file)

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nDtypes:")
print(df.dtypes)