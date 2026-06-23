import numpy as np
import pandas as pd
from typing import Dict, Optional

FEATURE_NAMES = [
    "longitude", "latitude", "velocity",
    "geo_altitude", "true_track", "vertical_rate"
]

ATTACK_TYPES = {
    "constant_position": "Constant Position Deviation Attack",
    "random_position": "Random Position Deviation Attack",
    "velocity_drift": "Velocity Drift Attack",
    "dos_deletion": "DoS / Message Deletion Attack",
    "flight_merge": "Flight Replacement / Merge Attack",
    "heading_manipulation": "Heading Manipulation Attack",
}


def generate_attack(
    df: pd.DataFrame,
    attack_type: str,
    icao24_target: Optional[str] = None,
    fraction: float = 0.3,
    seed: int = 42
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = df.copy()

    for c in FEATURE_NAMES:
        if c in df.columns:
            df[c] = df[c].astype(float)

    if icao24_target:
        target_mask = df["icao24"] == icao24_target
    else:
        target_icaos = df["icao24"].dropna().unique()
        n_select = max(1, int(len(target_icaos) * fraction))
        chosen = rng.choice(target_icaos, size=min(len(target_icaos), n_select), replace=False)
        target_mask = df["icao24"].isin(chosen)

    idx = df.index[target_mask]
    n = len(idx)

    if n == 0:
        df["is_anomaly"] = False
        df["anomaly_type"] = None
        return df

    if attack_type == "dos_deletion":
        keep_rows = df.index.difference(idx)
        result = df.loc[keep_rows].copy()
        result["is_anomaly"] = False
        result["anomaly_type"] = None
        return result

    df["is_anomaly"] = False
    df["anomaly_type"] = None
    df.loc[idx, "is_anomaly"] = True
    df.loc[idx, "anomaly_type"] = attack_type

    if attack_type == "constant_position":
        df.loc[idx, "latitude"] = df.loc[idx, "latitude"].values + 0.5
        df.loc[idx, "longitude"] = df.loc[idx, "longitude"].values + 0.5

    elif attack_type == "random_position":
        df.loc[idx, "latitude"] = df.loc[idx, "latitude"].values + rng.normal(0, 0.3, size=n)
        df.loc[idx, "longitude"] = df.loc[idx, "longitude"].values + rng.normal(0, 0.3, size=n)

    elif attack_type == "velocity_drift":
        base = df.loc[idx, "velocity"].values.astype(float)
        drift = np.linspace(0, 50, n) * rng.choice([-1, 1])
        df.loc[idx, "velocity"] = np.clip(base + drift, 0, 300)

    elif attack_type == "flight_merge":
        icao_set = set(df.loc[idx, "icao24"].unique())
        other_icaos = [i for i in df["icao24"].unique() if i not in icao_set]
        if other_icaos:
            donor_data = df[df["icao24"] == other_icaos[0]].head(n)
            if len(donor_data) > 0:
                for col in ["latitude", "longitude", "velocity", "geo_altitude"]:
                    df.loc[idx, col] = donor_data[col].values[:n]

    elif attack_type == "heading_manipulation":
        offset = rng.choice([-90, 90, 180], size=n)
        df.loc[idx, "true_track"] = (df.loc[idx, "true_track"].values + offset) % 360

    return df


def inject_all_attacks(
    df: pd.DataFrame,
    fractions: Optional[Dict[str, float]] = None,
    seed: int = 42
) -> pd.DataFrame:
    if fractions is None:
        fractions = {k: 0.1 for k in ATTACK_TYPES}

    df = df.copy()
    for c in FEATURE_NAMES:
        if c in df.columns:
            df[c] = df[c].astype(float)

    df["is_anomaly"] = False
    df["anomaly_type"] = None

    current = df.copy()

    for attack_type, fraction in fractions.items():
        clean_mask = current["is_anomaly"] == False
        clean_rows = current[clean_mask]

        if len(clean_rows) == 0:
            break

        attacked = generate_attack(clean_rows, attack_type, fraction=fraction, seed=seed)
        attacked = attacked.reset_index(drop=True)

        anomaly_sofar = current[current["is_anomaly"] == True].copy()
        current = pd.concat([anomaly_sofar, attacked], ignore_index=True)
        seed += 1

    return current
