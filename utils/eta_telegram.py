import os
import requests
from datetime import datetime, timedelta

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


WIB = timedelta(hours=7)

def ts_to_wib(ts):
    if ts is None:
        return "?"
    try:
        dt = datetime.fromtimestamp(ts) + WIB
        return dt.strftime("%d/%m/%Y %H:%M:%S WIB")
    except (OSError, ValueError, OverflowError):
        return str(ts)


def iso_to_wib(iso):
    if not iso:
        return "?"
    try:
        cleaned = iso.rstrip("Z")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt + WIB
        else:
            dt = dt + WIB
        return dt.strftime("%d/%m/%Y %H:%M:%S WIB")
    except (ValueError, TypeError):
        return iso


def send_eta_prediction(result: dict):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    if result.get("status") != "ok":
        return

    icao24 = result.get("icao24", "?")
    callsign = result.get("callsign") or "-"
    dest = result.get("destination") or "?"
    method = result.get("prediction_method") or "?"
    confidence = result.get("confidence") or 0
    eta_min = result.get("eta_minutes")
    eta_min_str = f"{eta_min:.1f}" if eta_min else "N/A"
    eta_sec = result.get("eta_seconds")
    eta_sec_str = f"({eta_sec} detik)" if eta_sec else ""
    dist = result.get("distance_km_to_dest")
    dist_str = f"{dist:.1f} km" if dist else "N/A"
    track_pts = result.get("track_points", "?")
    pos = result.get("current_position") or {}
    lat = pos.get("lat", "?")
    lon = pos.get("lon", "?")
    alt = pos.get("altitude", "?")
    speed = pos.get("speed_kmh", "?")
    heading = pos.get("heading", "?")
    eta_method = result.get("eta_method") or "?"
    route = result.get("route") or ""
    on_ground = result.get("on_ground", False)
    ground_str = "DI DARAT" if on_ground else "Terbang"
    predicted_at = iso_to_wib(result.get("predicted_at"))
    last_contact = ts_to_wib(result.get("last_contact"))

    message = (
        f"\U0001f680 {callsign} ({icao24}) \u2192 {dest}\n"
        f"\u251c\u2500 Method: {method} (conf: {confidence:.2f})\n"
        f"\u251c\u2500 ETA: {eta_min_str} menit {eta_sec_str}\n"
        f"\u251c\u2500 Jarak: {dist_str} | ETA Method: {eta_method}\n"
        f"\u251c\u2500 Posisi: {lat}, {lon}\n"
        f"\u251c\u2500 Alt: {alt}m | Speed: {speed} km/h | Heading: {heading}\U000000b0\n"
        f"\u251c\u2500 Track: {track_pts} points\n"
        f"\u251c\u2500 Kontak: {last_contact}\n"
        f"\u251c\u2500 Prediksi: {predicted_at}\n"
        f"\u2514\u2500 Status: {ground_str} | {result.get('status')}"
    )
    if route:
        message += f"\n  Route: {route}"

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=5)
        print(f"[TELEGRAM] Notif sent for {callsign} ({icao24})")
    except Exception as e:
        print(f"[TELEGRAM] Send failed: {e}")
