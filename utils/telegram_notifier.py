import os
import time
import requests
from datetime import datetime


def _env(key, default=""):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    if k.strip() == key:
                        return v.strip()
    return default


DEV_TOKEN = _env("TELEGRAM_BOT_TOKEN_DEV")
DEV_CHAT_ID = _env("TELEGRAM_CHAT_DEV")
STAKE_TOKEN = _env("TELEGRAM_BOT_TOKEN_STAKE")
STAKE_CHAT_ID = _env("TELEGRAM_CHAT_STAKE")

_sentinel = {"blocked": False, "last_attempt": 0.0, "backoff": 60}
_anomaly_cooldown = {}


def _send(token, chat_id, text):
    if not token or not chat_id:
        return False
    now = time.time()
    if _sentinel["blocked"] and now - _sentinel["last_attempt"] < _sentinel["backoff"]:
        return False
    _sentinel["last_attempt"] = now
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        }, timeout=10)
        if resp.status_code == 200:
            _sentinel["blocked"] = False
            _sentinel["backoff"] = 60
            return True
        _sentinel["blocked"] = True
        _sentinel["backoff"] = min(_sentinel["backoff"] * 2, 600)
        return False
    except:
        _sentinel["blocked"] = True
        _sentinel["backoff"] = min(_sentinel["backoff"] * 2, 600)
        return False


def notify_performance(stats_dict, model_name="VAE-LSTM", p50=None, p90=None, p95=None, max_err=None, uptime=None):
    ts = datetime.now().strftime("%H:%M:%S")
    total = stats_dict.get("total", 0)
    windows = stats_dict.get("buffer_ready", 0)
    anomalies = stats_dict.get("anomalies", 0)
    es_errors = stats_dict.get("es_errors", 0)
    reconnects = stats_dict.get("reconnects", 0)
    anom_pct = (anomalies / max(windows, 1)) * 100
    attack_counts = stats_dict.get("attack_counts", {})

    uptime_str = ""
    if uptime is not None:
        mins = int(uptime // 60)
        secs = int(uptime % 60)
        uptime_str = f"{mins}m {secs}s"

    p50_s = f"{p50:.2f}" if p50 is not None else "N/A"
    p90_s = f"{p90:.2f}" if p90 is not None else "N/A"
    p95_s = f"{p95:.2f}" if p95 is not None else "N/A"
    max_s = f"{max_err:.2f}" if max_err is not None else "N/A"
    thr = f"{total / max(uptime or 1, 1):.0f}"

    attack_lines = ""
    for at in ["dos_deletion", "heading_manipulation", "velocity_drift", "flight_merge", "constant_position", "random_position"]:
        cnt = attack_counts.get(at, 0)
        if cnt > 0:
            attack_lines += f"\n  {at:.<20} : {cnt}"
    if not attack_lines:
        attack_lines = "\n  (none)"

    text = (
        f"<b>━━━ {model_name} — Performance Matrix ━━━</b>\n"
        f"Time: {ts}  |  Uptime: {uptime_str}\n"
        f"─────────────────────────────────────────\n"
        f"<b>Anomaly Metrics</b>\n"
        f"  Windows processed .. : {windows}\n"
        f"  Anomalies .......... : {anomalies} ({anom_pct:.1f}%)\n"
        f"─────────────────────────────────────────\n"
        f"<b>Error Distribution</b>\n"
        f"  p50/p90/p95 ....... : {p50_s} / {p90_s} / {p95_s}\n"
        f"  Max ............... : {max_s}\n"
        f"─────────────────────────────────────────\n"
        f"<b>System Health</b>\n"
        f"  Throughput ........ : {thr} msg/s\n"
        f"  ES Errors ......... : {es_errors}\n"
        f"  Reconnects ........ : {reconnects}\n"
        f"─────────────────────────────────────────\n"
        f"<b>Attack Types</b>{attack_lines}\n"
        f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
    )
    return _send(DEV_TOKEN, DEV_CHAT_ID, text)


def notify_anomaly(flight_info, model_name="VAE-LSTM", cooldown=10, threshold=None):
    icao24 = flight_info.get("icao24", "unknown")
    now = time.time()
    last = _anomaly_cooldown.get(icao24, 0)
    if now - last < cooldown:
        return False
    _anomaly_cooldown[icao24] = now

    ts = datetime.now().strftime("%H:%M:%S")
    err_val = flight_info.get("recon_error") or flight_info.get("score", "?")
    if isinstance(err_val, float):
        err_val = f"{err_val:.2f}"
    lat = flight_info.get("latitude", "?")
    lon = flight_info.get("longitude", "?")
    callsign = flight_info.get("callsign", "?")
    attack_type = flight_info.get("attack_type", "?")
    vel = flight_info.get("velocity", "?")
    alt = flight_info.get("baro_altitude", "?")

    thr_str = f" (threshold: {threshold})" if threshold is not None else ""

    text_dev = (
        f"<b>━━━ ⚠️ ANOMALI — {model_name} ━━━</b>\n"
        f"Pesawat : {callsign} ({icao24})\n"
        f"Waktu .. : {ts}\n"
        f"Jenis .. : {attack_type}\n"
        f"Error .. : {err_val}{thr_str}\n"
        f"Posisi . : {lat}°N, {lon}°W\n"
        f"Speed .. : {vel} kts  |  Alt : {alt} ft"
    )
    text_stake = (
        f"<b>━━━ ⚠️ ANOMALI PENERBANGAN ━━━</b>\n"
        f"Pesawat : {callsign}\n"
        f"Waktu .. : {ts}\n"
        f"Lokasi . : {lat}°N, {lon}°W\n"
        f"Speed .. : {vel} kts  |  Alt : {alt} ft\n"
        f"Keterangan : Terdeteksi anomali ({attack_type})"
    )
    _send(DEV_TOKEN, DEV_CHAT_ID, text_dev)
    _send(STAKE_TOKEN, STAKE_CHAT_ID, text_stake)


def notify_startup(model_name="VAE-LSTM"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text_dev = f"<b>ADS-B System Online</b>\nModel: {model_name}\nStarted: {ts}"
    text_stake = f"<b>Sistem Monitoring Penerbangan Aktif</b>\nMulai: {ts}"
    _send(DEV_TOKEN, DEV_CHAT_ID, text_dev)
    _send(STAKE_TOKEN, STAKE_CHAT_ID, text_stake)
