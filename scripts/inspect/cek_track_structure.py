import pandas as pd

df = pd.read_parquet("data/historical/eta_training_1724544000.parquet")

print(df.iloc[0][[
    "estdepartureairport",
    "estarrivalairport"
]])