#!/usr/bin/env python3
"""
Predict destination airport & ETA from real-time flight track.

Usage:
    python predict_destination.py <icao24>
    python predict_destination.py --demo
"""

import sys
import json
from eta_pipeline import predict, demo, make_track


def predict_from_es(icao24):
    result = predict(icao24)
    return result


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    if sys.argv[1] == "--demo":
        demo()
        return

    if sys.argv[1] == "--manual":
        # manual input
        try:
            lat = float(input("Latitude: "))
            lon = float(input("Longitude: "))
            alt = float(input("Altitude (m): "))
            hdg = float(input("Heading (deg): "))
            speed = float(input("Speed (km/h): "))
            callsign = input("Callsign (optional): ").strip() or None
        except ValueError:
            print("Invalid input")
            return

        track = make_track(lat, lon, alt, hdg, speed, callsign)
        result = predict(icao24="manual", track=track)
    else:
        icao24 = sys.argv[1]
        result = predict_from_es(icao24)

    print("\n===== PREDICTION RESULT =====")
    for key, value in result.items():
        if key == "heading_top5":
            continue
        print(f"{key}: {value}")

    if "heading_top5" in result:
        print("\nTop 5 heading-scored airports:")
        for r in result["heading_top5"]:
            airport_name = None
            try:
                import pandas as pd
                airports = pd.read_csv("data/final/airport_lookup.csv")
                match = airports[airports["icao"] == r["airport"]]
                if len(match) > 0:
                    airport_name = match.iloc[0]["airport_name"]
            except Exception:
                pass
            name_str = f" ({airport_name})" if airport_name else ""
            print(f"  {r['airport']}{name_str}: {r['score']:.4f}")

    print(f"\nJSON output:")
    print(json.dumps({k: v for k, v in result.items() if k != "heading_top5"}, indent=2))


if __name__ == "__main__":
    main()
