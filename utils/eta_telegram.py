import os
import csv
import requests
from datetime import datetime, timedelta

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

WIB = timedelta(hours=7)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIRPORT_FILE = os.path.join(BASE_DIR, "data", "final", "airport_lookup.csv")

airport_names = {}
try:
    with open(AIRPORT_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            airport_names[row["icao"]] = row.get("airport_name", row["icao"])
except Exception:
    pass

AIRLINE_PREFIXES = {
    "AUA": "Austrian Airlines",
    "BAW": "British Airways",
    "DLH": "Lufthansa",
    "EJU": "easyJet",
    "ETH": "Ethiopian Airlines",
    "FIN": "Finnair",
    "GEC": "Lufthansa Cargo",
    "IBE": "Iberia",
    "KLM": "KLM",
    "PGT": "Pegasus Airlines",
    "QTR": "Qatar Airways",
    "SAS": "Scandinavian Airlines",
    "THY": "Turkish Airlines",
    "VLG": "Vueling",
    "WZZ": "Wizz Air",
    "AEA": "Air Europa",
    "ANE": "Air Nostrum",
    "RYR": "Ryanair",
    "EZY": "easyJet",
}

WIB = timedelta(hours=7)


def get_airport_name(icao):
    if not icao:
        return None
    name = airport_names.get(icao)
    return name if name else None


def get_airline(callsign):
    if not callsign or callsign == "-":
        return None
    prefix = callsign[:3]
    return AIRLINE_PREFIXES.get(prefix)


def format_airport(icao):
    name = get_airport_name(icao)
    if name:
        return f"{name} ({icao})"
    return icao or "?"


def ts_to_wib(ts):
    if ts is None:
        return "?"
    try:
        dt = datetime.fromtimestamp(ts) + WIB
        return dt.strftime("%H:%M WIB")
    except (OSError, ValueError, OverflowError):
        return str(ts)


def iso_to_wib(iso):
    if not iso:
        return "?"
    try:
        cleaned = iso.rstrip("Z")
        dt = datetime.fromisoformat(cleaned)
        return (dt + WIB).strftime("%H:%M WIB")
    except (ValueError, TypeError):
        return iso


METHOD_LABELS = {
    "callsign": "Route Lookup",
    "ml_classifier": "ML Classifier",
    "heading_scoring": "Heading Score",
}


def send_eta_prediction(result: dict):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    if result.get("status") != "ok":
        return

    icao24 = result.get("icao24", "?")
    callsign = result.get("callsign") or "-"
    destination = result.get("destination") or "?"
    origin = result.get("origin")
    eta_seconds = result.get("eta_seconds")
    eta_minutes = result.get("eta_minutes")
    on_ground = result.get("on_ground", False)
    predicted_at_raw = result.get("predicted_at")
    last_contact_raw = result.get("last_contact")
    confidence = result.get("confidence")
    method = result.get("prediction_method")

    # Nama maskapai
    airline = get_airline(callsign)
    airline_str = f" ({airline})" if airline else ""

    # Nama bandara
    dest_str = format_airport(destination)
    origin_str = ""
    if origin:
        origin_str = f"Dari: {format_airport(origin)} \u2192 "

    # Estimasi jam tiba = predicted_at + eta_seconds
    arrival_str = ""
    eta_remaining_str = ""
    if predicted_at_raw and eta_seconds:
        try:
            cleaned = predicted_at_raw.rstrip("Z")
            dt_pred = datetime.fromisoformat(cleaned)
            dt_arrival = dt_pred + timedelta(seconds=eta_seconds)
            arrival_str = (dt_arrival + WIB).strftime("%H:%M WIB")
            if eta_minutes:
                eta_remaining_str = f" | Sisa: {eta_minutes:.0f} menit"
        except (ValueError, TypeError):
            arrival_str = "?"

    # Confidence + Method
    conf_str = ""
    if confidence is not None:
        method_str = METHOD_LABELS.get(method, method or "?")
        conf_str = f"\n   \U0001f4ca Confidence: {confidence*100:.0f}% ({method_str})"

    # Status
    status_str = "\U00002705 Terbang" if not on_ground else "\U0001f6ec Mendarat"

    # Update terakhir
    time_str = ""
    if predicted_at_raw:
        time_str = iso_to_wib(predicted_at_raw)
    elif last_contact_raw:
        time_str = ts_to_wib(last_contact_raw)

    message = (
        f"\U0001f6ec {callsign}{airline_str}\n"
        f"   {origin_str}Ke: {dest_str}\n"
        f"   \U0001f550 Tiba: {arrival_str}{eta_remaining_str}"
        f"{conf_str}\n"
        f"   {status_str}"
    )
    if time_str:
        message += f" | Update: {time_str}"

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=5)
        print(f"[TELEGRAM] Notif sent for {callsign}")
    except Exception as e:
        print(f"[TELEGRAM] Send failed: {e}")


def send_landing_alert(result: dict):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    if result.get("status") != "ok":
        return

    callsign = result.get("callsign") or "-"
    destination = result.get("destination") or "?"
    origin = result.get("origin")
    eta_minutes = result.get("eta_minutes")
    cp = result.get("current_position", {})
    dist = result.get("distance_km_to_dest")
    alt = cp.get("altitude") if isinstance(cp, dict) else None
    spd = cp.get("speed_kmh") if isinstance(cp, dict) else None

    airline = get_airline(callsign)
    airline_str = f" ({airline})" if airline else ""
    dest_str = format_airport(destination)
    origin_str = ""
    if origin:
        origin_str = f"Dari: {format_airport(origin)} \u2192 "

    info_parts = []
    if dist is not None:
        info_parts.append(f"\U0001f4cf Jarak: {dist:.0f} km")
    if alt is not None:
        info_parts.append(f"\U0001f4d0 Alt: {alt:.0f} ft")
    if spd is not None:
        info_parts.append(f"\U0001f4a8 {spd:.0f} km/h")
    info_str = " | ".join(info_parts)

    eta_str = ""
    if eta_minutes is not None:
        eta_str = f"\n   \U0001f550 ETA: ~{eta_minutes:.0f} menit lagi"

    message = (
        f"\U0001f6ec LANDING IMMINENT\n"
        f"   {callsign}{airline_str}\n"
        f"   {origin_str}Ke: {dest_str}\n"
        f"   {info_str}{eta_str}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=5)
        print(f"[TELEGRAM] Landing alert sent for {callsign}")
    except Exception as e:
        print(f"[TELEGRAM] Landing alert failed: {e}")
