import os
import time
import sys
import math
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch

ES_HOST = os.getenv("ES_HOST", "http://127.0.0.1:9200")
INDEX_NAME = "flights"
BATCH_SIZE = 50

es = Elasticsearch(ES_HOST)
try:
    es.info()
    print(f"ES connected: {ES_HOST}")
except Exception as e:
    print(f"ES error: {e}")
    sys.exit(1)

from eta_pipeline_lgbm import (
    predict_by_callsign,
    predict_by_lgbm,
    predict_by_heading,
    predict_eta,
    get_trajectory,
    airport_dict,
    callsign_lookup,
    haversine,
    bearing,
)

def find_landing_icaos(limit=200):
    body = {
        "query": {"term": {"on_ground": True}},
        "aggs": {
            "icaos": {
                "terms": {"field": "icao24.keyword", "size": limit, "order": {"max_ts": "desc"}},
                "aggs": {
                    "max_ts": {"max": {"field": "last_contact"}}
                }
            }
        },
        "size": 0
    }
    res = es.search(index=INDEX_NAME, body=body)
    icaos = []
    for b in res["aggregations"]["icaos"]["buckets"]:
        icaos.append((b["key"], int(b["max_ts"]["value"])))
    return icaos


def get_trajectory_landing(icao24, size=60):
    body = {
        "query": {"term": {"icao24.keyword": icao24}},
        "sort": [{"timestamp": {"order": "desc"}}],
        "size": size
    }
    try:
        res = es.search(index=INDEX_NAME, body=body)
    except Exception:
        return pd.DataFrame()
    rows = []
    for hit in res["hits"]["hits"]:
        s = hit["_source"]
        rows.append({
            "timestamp": s.get("timestamp", 0),
            "lat": s.get("latitude"),
            "lon": s.get("longitude"),
            "alt": s.get("baro_altitude") or s.get("geo_altitude") or 0,
            "on_ground": s.get("on_ground", False),
            "velocity": s.get("velocity_kmh") or 0,
            "heading": s.get("true_track") or 0,
            "callsign": s.get("callsign"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def find_nearest_airport(lat, lon):
    best_dist = float("inf")
    best_ap = None
    for ap, (ap_lat, ap_lon) in airport_dict.items():
        d = haversine(lat, lon, ap_lat, ap_lon)
        if d < best_dist:
            best_dist = d
            best_ap = ap
    return best_ap, best_dist


def evaluate():
    print("=" * 70)
    print("  LIGHTGBM EVALUATION (via Landing Events)")
    print("=" * 70)

    print("\n[1] Scanning for landing flights...")
    landing_icaos = find_landing_icaos(300)
    print(f"     Found {len(landing_icaos)} potential landing ICAOs")

    landings = []
    for icao, last_ts in landing_icaos:
        traj = get_trajectory_landing(icao, size=60)
        if len(traj) < 5:
            continue

        ground_idx = None
        for i in range(len(traj) - 1, -1, -1):
            if traj.iloc[i]["on_ground"]:
                ground_idx = i
            elif ground_idx is not None:
                break

        if ground_idx is None or ground_idx == 0:
            continue

        last_air_idx = ground_idx - 1
        if last_air_idx < 0:
            continue

        last_air = traj.iloc[last_air_idx]

        alt = last_air["alt"]
        lat = last_air["lat"]
        lon = last_air["lon"]

        if alt < 300 or alt > 5000:
            continue

        nearest_ap, dist_km = find_nearest_airport(lat, lon)
        if nearest_ap is None or dist_km > 50:
            continue

        predict_idx = None
        for i in range(last_air_idx - 1, -1, -1):
            ts_diff = last_air["timestamp"] - traj.iloc[i]["timestamp"]
            if ts_diff >= 300:
                predict_idx = i
                break
        if predict_idx is None:
            predict_idx = 0

        pred_pt = traj.iloc[predict_idx]

        actual_eta = last_air["timestamp"] - pred_pt["timestamp"]
        if actual_eta < 60 or actual_eta > 72000:
            continue

        landings.append({
            "icao24": icao,
            "destination_actual": nearest_ap,
            "dist_to_airport_km": round(dist_km, 1),
            "pred_lat": pred_pt["lat"],
            "pred_lon": pred_pt["lon"],
            "pred_alt": pred_pt["alt"],
            "pred_hdg": pred_pt["heading"],
            "pred_speed": pred_pt["velocity"],
            "pred_callsign": pred_pt.get("callsign"),
            "actual_eta_seconds": actual_eta,
        })

        if len(landings) >= 150:
            break

    print(f"     => {len(landings)} valid landing events found\n")

    if len(landings) < 10:
        print("  Too few samples. Try increasing scan range.")
        return

    print("-" * 70)
    print("  DESTINATION PREDICTION")
    print("-" * 70)

    dest_results = {
        "lgbm_classifier": {"correct": 0, "total": 0, "time": 0},
        "heading_scoring": {"correct": 0, "total": 0, "time": 0, "top5_correct": 0},
        "callsign": {"total": 0, "time": 0},
    }

    for ld in landings:
        lat, lon = ld["pred_lat"], ld["pred_lon"]
        alt, hdg = ld["pred_alt"], ld["pred_hdg"]
        gt = ld["destination_actual"]

        if alt < 100:
            continue

        t0 = time.time()
        lgbm_result = predict_by_lgbm(lat, lon, alt, hdg)
        dest_results["lgbm_classifier"]["time"] += time.time() - t0
        dest_results["lgbm_classifier"]["total"] += 1
        if lgbm_result and lgbm_result["destination"] == gt:
            dest_results["lgbm_classifier"]["correct"] += 1

        t0 = time.time()
        hdg_results = predict_by_heading(lat, lon, hdg, alt)
        dest_results["heading_scoring"]["time"] += time.time() - t0
        dest_results["heading_scoring"]["total"] += 1
        if hdg_results and hdg_results[0]["airport"] == gt:
            dest_results["heading_scoring"]["correct"] += 1
        if hdg_results:
            top5 = [r["airport"] for r in hdg_results[:5]]
            if gt in top5:
                dest_results["heading_scoring"]["top5_correct"] += 1

        t0 = time.time()
        cs = ld["pred_callsign"]
        cs_result = predict_by_callsign(cs)
        dest_results["callsign"]["time"] += time.time() - t0
        if cs_result:
            dest_results["callsign"]["total"] += 1

    n_dest = dest_results["lgbm_classifier"]["total"]
    print(f"\n  Samples: {n_dest}")
    print()

    for method, label in [("lgbm_classifier", "lgbm_classifier"),
                           ("heading_scoring", "heading_scoring (top-1)")]:
        r = dest_results[method]
        acc = r["correct"] / r["total"] * 100 if r["total"] else 0
        avg_t = r["time"] / r["total"] * 1000 if r["total"] else 0
        print(f"  {label:30s}  {acc:5.1f}%  ({r['correct']}/{r['total']})  {avg_t:.1f}ms")

    r = dest_results["heading_scoring"]
    top5_recall = r["top5_correct"] / r["total"] * 100 if r["total"] else 0
    print(f"  {'heading_scoring (top-5)':30s}  {top5_recall:5.1f}%  ({r['top5_correct']}/{r['total']})")

    r = dest_results["callsign"]
    match_rate = r["total"] / n_dest * 100 if n_dest else 0
    print(f"  {'callsign match rate':30s}  {match_rate:5.1f}%  ({r['total']}/{n_dest})")

    print()
    print("  Method distribution (simulated pipeline):")
    method_usage = {"lgbm_classifier": 0, "callsign": 0, "heading_scoring": 0, "failed": 0}
    for ld in landings:
        lat, lon = ld["pred_lat"], ld["pred_lon"]
        alt, hdg = ld["pred_alt"], ld["pred_hdg"]
        cs = ld["pred_callsign"]

        if alt < 100:
            method_usage["failed"] += 1
            continue

        if cs:
            cs_r = predict_by_callsign(cs)
            if cs_r:
                method_usage["callsign"] += 1
                continue

        lgbm_r = predict_by_lgbm(lat, lon, alt, hdg)
        if lgbm_r:
            method_usage["lgbm_classifier"] += 1
            continue

        hdg_r = predict_by_heading(lat, lon, hdg, alt)
        if hdg_r:
            method_usage["heading_scoring"] += 1
            continue

        method_usage["failed"] += 1

    total = sum(method_usage.values())
    for m in ["lgbm_classifier", "callsign", "heading_scoring", "failed"]:
        pct = method_usage[m] / total * 100 if total else 0
        print(f"    {m:20s}  {method_usage[m]:5d} ({pct:.1f}%)")

    print()
    print("-" * 70)
    print("  ETA PREDICTION")
    print("-" * 70)

    eta_results = {"distance_speed": [], "lgbm": [], "route_avg": []}
    eta_samples = 0

    for ld in landings:
        lat, lon = ld["pred_lat"], ld["pred_lon"]
        alt, hdg = ld["pred_alt"], ld["pred_hdg"]
        speed = ld["pred_speed"]
        gt_dest = ld["destination_actual"]
        true_eta = ld["actual_eta_seconds"]

        if speed < 50 or alt < 100:
            continue

        for elapsed in [None, 300]:
            eta_r = predict_eta(lat, lon, gt_dest, speed, alt, hdg, elapsed)
            method = eta_r.get("eta_method", "none")
            pred = eta_r.get("eta_seconds")
            if pred and method in eta_results and pred > 0:
                error = abs(pred - true_eta)
                eta_results[method].append(error)

        eta_samples += 1
        if eta_samples >= 100:
            break

    print(f"\n  Samples: {eta_samples}")
    print(f"  Baseline: actual remaining time until landing\n")

    for method in ["distance_speed", "lgbm", "route_avg"]:
        errs = eta_results[method]
        if errs:
            mae = np.mean(errs) / 60
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) / 60
            print(f"  {method:20s}  MAE={mae:6.1f} min  RMSE={rmse:6.1f} min  n={len(errs)}")
        else:
            print(f"  {method:20s}  (no data)")

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    lgbm_acc = dest_results["lgbm_classifier"]["correct"] / max(dest_results["lgbm_classifier"]["total"], 1) * 100
    hdg_acc = dest_results["heading_scoring"]["correct"] / max(dest_results["heading_scoring"]["total"], 1) * 100

    print(f"""
  Dataset  : {n_dest} landing events from {len(landing_icaos)} scanned ICAOs

  Destination Accuracy:
    lgbm_classifier     : {lgbm_acc:.1f}%
    heading_scoring     : {hdg_acc:.1f}% (top-1), {top5_recall:.1f}% (top-5)

  Coverage (method used):
    lgbm_classifier     : {method_usage['lgbm_classifier']/total*100:.1f}%
    heading_scoring     : {method_usage['heading_scoring']/total*100:.1f}%
    callsign            : {method_usage['callsign']/total*100:.1f}%
""")

    print("  ETA Error (MAE):")
    for method in ["distance_speed", "lgbm", "route_avg"]:
        errs = eta_results[method]
        if errs:
            print(f"    {method:20s}  {np.mean(errs)/60:.1f} min")
    print()


if __name__ == "__main__":
    evaluate()
