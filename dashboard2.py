"""
SkyWatch — Real-Time Air Traffic Monitoring Dashboard (v2 - Real API)
======================================================================
Dashboard pemantauan lalu lintas udara real-time yang mengintegrasikan
data prediksi ETA (ETA API) dan deteksi anomali penerbangan (Anomaly API).

Cara menjalankan:
    streamlit run dashboard2.py

Konfigurasi endpoint backend:
    ETA_API_HOST (default: http://100.94.21.31:8002)
    ANOMALY_API_HOST (default: http://100.126.247.116:8001)
"""

import os
import math
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

# ──────────────────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyWatch — Air Traffic Monitor",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)

ETA_API_HOST = os.getenv("ETA_API_HOST", "http://100.94.21.31:8002")
ANOMALY_API_HOST = os.getenv("ANOMALY_API_HOST", "http://100.126.247.116:8001")
AUTO_REFRESH_SEC = int(os.getenv("AUTO_REFRESH_SEC", "15"))

METHODS = ["Route Lookup", "ML Classifier", "Heading Score"]
METHOD_COLORS = {"callsign": "#38bdf8", "ml_classifier": "#34d399", "heading_scoring": "#fbbf24"}
METHOD_LABELS = {"callsign": "Route Lookup", "ml_classifier": "ML Classifier", "heading_scoring": "Heading Score"}

# ──────────────────────────────────────────────────────────────────────────
# THEME / CSS
# ──────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

:root {
    --sw-bg: #0b1220;
    --sw-panel: #121a2b;
    --sw-panel-2: #16213a;
    --sw-border: #233252;
    --sw-accent: #38bdf8;
    --sw-accent-2: #22d3ee;
    --sw-green: #34d399;
    --sw-amber: #fbbf24;
    --sw-red: #f87171;
    --sw-text: #e2e8f0;
    --sw-muted: #8aa0c4;
}

.stApp {
    background: radial-gradient(circle at 10% 0%, #0e1a30 0%, #0b1220 45%, #070b14 100%);
    color: var(--sw-text);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a0f1c 100%);
    border-right: 1px solid var(--sw-border);
}

h1, h2, h3, h4 { color: #f1f5f9 !important; letter-spacing: -0.02em; }

.sw-hero {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 26px; border-radius: 18px; margin-bottom: 18px;
    background: linear-gradient(120deg, rgba(56,189,248,0.12), rgba(34,211,238,0.04) 60%, transparent);
    border: 1px solid var(--sw-border);
}
.sw-hero h1 { font-size: 1.9rem; font-weight: 800; margin: 0; }
.sw-hero p { color: var(--sw-muted); margin: 2px 0 0 0; font-size: 0.92rem; }
.sw-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    border: 1px solid var(--sw-border); background: rgba(255,255,255,0.03);
}
.sw-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.sw-dot.online { background: var(--sw-green); box-shadow: 0 0 8px var(--sw-green); }
.sw-dot.offline { background: var(--sw-red); box-shadow: 0 0 8px var(--sw-red); }
.sw-dot.demo { background: var(--sw-amber); box-shadow: 0 0 8px var(--sw-amber); }

.metric-card {
    background: linear-gradient(160deg, var(--sw-panel) 0%, var(--sw-panel-2) 100%);
    border: 1px solid var(--sw-border);
    border-radius: 16px;
    padding: 16px 18px;
    height: 100%;
    transition: transform .15s ease, border-color .15s ease;
}
.metric-card:hover { transform: translateY(-2px); border-color: var(--sw-accent); }
.metric-label { color: var(--sw-muted); font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
.metric-value { font-size: 1.85rem; font-weight: 800; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
.metric-sub { font-size: 0.78rem; margin-top: 4px; font-weight: 600; }
.metric-icon { font-size: 1.3rem; opacity: .85; }

.sw-panel {
    background: var(--sw-panel);
    border: 1px solid var(--sw-border);
    border-radius: 16px;
    padding: 18px 20px;
}

.phase-badge {
    display: inline-block; padding: 4px 14px; border-radius: 999px;
    font-weight: 700; font-size: 0.8rem; letter-spacing: .04em;
    border: 1px solid var(--sw-border);
}

.flight-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 12px; border-radius: 10px; margin-bottom: 6px;
    background: rgba(255,255,255,0.02); border: 1px solid var(--sw-border);
}
.flight-row:hover { background: rgba(56,189,248,0.06); }

.anomaly-pill {
    padding: 3px 10px; border-radius: 999px; font-size: 0.72rem; font-weight: 700;
}
.sev-low { background: rgba(52,211,153,0.15); color: #34d399; }
.sev-med { background: rgba(251,191,36,0.15); color: #fbbf24; }
.sev-high { background: rgba(248,113,113,0.18); color: #f87171; }

div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--sw-panel); padding: 6px; border-radius: 14px; border: 1px solid var(--sw-border);}
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 10px 16px; font-weight: 600; color: var(--sw-muted); }
.stTabs [aria-selected="true"] { background: linear-gradient(120deg, var(--sw-accent), var(--sw-accent-2)); color: #04101f !important; }

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* Paksa warna teks putih untuk readability di background gelap */
.stApp p, .stApp span, .stApp div, .stApp label, .stApp .st-emotion-cache-1dj0hjr,
.stApp .st-emotion-cache-10trblm, .stApp .st-emotion-cache-1mi2ry5,
div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"],
.stTextInput label, .st-bp, .st-bq, .st-cb, .st-c9,
[data-testid="stMetricValue"] > div, [data-testid="baseButton-header"],
.stDataFrame, [data-testid="StyledDataFrameColHeader"] { color: #e2e8f0 !important; }

/* Selectbox: pakai tema gelap agar dropdown terbaca (teks putih di latar gelap) */
.stSelectbox label, .stSlider label, .stMultiSelect label { color: #e2e8f0 !important; }
.stSelectbox [data-baseweb="select"] > div, .stSelectbox input,
.stSelectbox div[role="combobox"] { color: #e2e8f0 !important; background: var(--sw-panel) !important; }
.stSelectbox li, .stSelectbox [role="option"], 
.stSelectbox [data-baseweb="menu"] div, 
.stSelectbox [data-baseweb="popover"] div,
div[data-baseweb="select"] ul li,
ul[role="listbox"] li,
div[role="listbox"] div { color: #e2e8f0 !important; background: var(--sw-panel-2) !important; }
ul[role="listbox"] li:hover,
div[role="listbox"] div:hover { background: rgba(255,255,255,0.04) !important; color: #e2e8f0 !important; }

.st-bw, .st-cg, .st-ch, .st-ci, .st-ae, .st-af, .st-ag { color: #e2e8f0 !important; }
.st-cx { color: #0b1220 !important; }
[data-testid="stExpander"] { border-color: #233252 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# AIRPORT / CALLSIGN LOOKUP
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_airport_lookup():
    path = os.path.join(os.path.dirname(__file__), "data", "final", "airport_lookup.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            result = {}
            for _, row in df.iterrows():
                icao = str(row.get("icao", "")).upper().strip()
                result[icao] = {
                    "name": row.get("airport_name", icao),
                    "country": row.get("country", ""),
                    "lat": row.get("lat", None),
                    "lon": row.get("lon", None),
                }
            return result
        except Exception:
            return {}
    return {}

@st.cache_data(ttl=3600, show_spinner=False)
def load_callsign_routes():
    path = os.path.join(os.path.dirname(__file__), "data", "final", "callsign_route_lookup.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            result = {}
            for _, row in df.iterrows():
                cs = str(row.get("callsign", "")).strip().upper()
                route = str(row.get("route", "")).strip()
                if cs and route and "_" in route:
                    parts = route.split("_")
                    result[cs] = {"origin": parts[0], "destination": parts[1]}
            return result
        except Exception:
            return {}
    return {}

airport_lookup = load_airport_lookup()
callsign_routes = load_callsign_routes()

def get_airport_info(icao_code):
    if not icao_code:
        return None
    return airport_lookup.get(str(icao_code).upper().strip())

def get_callsign_info(callsign):
    if not callsign:
        return None
    cs = str(callsign).strip().upper()
    info = callsign_routes.get(cs)
    if not info:
        info = callsign_routes.get(cs[:3])
    return info

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def bearing_deg(lat1, lon1, lat2, lon2):
    """Bearing dari pos1 ke pos2 dalam derajat (0-360)"""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon_r = math.radians(lon2 - lon1)
    x = math.sin(dlon_r) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r)
    b = math.degrees(math.atan2(x, y))
    return (b + 360) % 360

def heading_diff(h1, h2):
    d = abs(h1 - h2)
    return min(d, 360 - d)

def estimate_origin_from_heading(lat, lon, heading, dest_code=None, max_dist_km=800):
    """Estimasi bandara asal berdasarkan heading + destination.
    Cari bandara di belakang pesawat yang paling segaris dengan rute menuju destination."""
    if lat is None or lon is None or heading is None:
        return None
    dest_info = airport_lookup.get(dest_code) if dest_code else None
    candidates = []
    for code, info in airport_lookup.items():
        ap_lat = info.get("lat")
        ap_lon = info.get("lon")
        if ap_lat is None or ap_lon is None:
            continue
        dist = haversine_km(lat, lon, ap_lat, ap_lon)
        if dist > max_dist_km:
            continue
        bear = bearing_deg(lat, lon, ap_lat, ap_lon)
        hd = heading_diff(heading, bear)
        # Bandara ada di belakang: heading berlawanan (>135 deg)
        if hd <= 135:
            continue
        # Validasi via destination: semakin kecil sudut (airport→plane) vs (airport→dest), semakin baik
        score = dist
        if dest_info:
            dest_lat = dest_info.get("lat")
            dest_lon = dest_info.get("lon")
            if dest_lat is not None and dest_lon is not None:
                b_ap_to_plane = bearing_deg(ap_lat, ap_lon, lat, lon)
                b_ap_to_dest = bearing_deg(ap_lat, ap_lon, dest_lat, dest_lon)
                angle = heading_diff(b_ap_to_plane, b_ap_to_dest)
                # Kalau pesawat searah dengan route menuju dest, angle kecil → bagus
                score = dist * (1 + angle / 90)
        candidates.append((score, code))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]

# ──────────────────────────────────────────────────────────────────────────
# ETA API FETCHERS
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=AUTO_REFRESH_SEC, show_spinner=False)
def fetch_eta_stats():
    try:
        r = requests.get(f"{ETA_API_HOST}/eta/stats", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None

@st.cache_data(ttl=AUTO_REFRESH_SEC, show_spinner=False)
def fetch_eta_predictions():
    try:
        r = requests.get(f"{ETA_API_HOST}/eta/predictions?limit=500", timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None

@st.cache_data(ttl=AUTO_REFRESH_SEC, show_spinner=False)
def fetch_eta_geojson():
    try:
        r = requests.get(f"{ETA_API_HOST}/eta/predictions.geojson?limit=500", timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None

@st.cache_data(ttl=AUTO_REFRESH_SEC, show_spinner=False)
def fetch_eta_history(icao24, hours=3):
    try:
        r = requests.get(f"{ETA_API_HOST}/eta/history/{icao24}", params={"hours": hours, "limit": 1000}, timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────
# ANOMALY API FETCHERS
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=AUTO_REFRESH_SEC, show_spinner=False)
def fetch_anomaly_health():
    try:
        r = requests.get(f"{ANOMALY_API_HOST}/health", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None

@st.cache_data(ttl=AUTO_REFRESH_SEC, show_spinner=False)
def fetch_anomaly_stats():
    try:
        r = requests.get(f"{ANOMALY_API_HOST}/api/stats", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None

@st.cache_data(ttl=AUTO_REFRESH_SEC, show_spinner=False)
def fetch_anomaly_results(limit=500):
    try:
        r = requests.get(f"{ANOMALY_API_HOST}/api/results", params={"limit": limit}, timeout=10)
        return r.json() if r.ok else []
    except Exception:
        return []

@st.cache_data(ttl=AUTO_REFRESH_SEC, show_spinner=False)
def fetch_anomaly_by_type(attack_type="", limit=200):
    try:
        r = requests.get(f"{ANOMALY_API_HOST}/api/anomalies", params={"type": attack_type, "limit": limit}, timeout=10)
        return r.json() if r.ok else []
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────
# LOAD DATA (real API)
# ──────────────────────────────────────────────────────────────────────────
eta_stats = fetch_eta_stats()
eta_preds = fetch_eta_predictions()
eta_gj = fetch_eta_geojson()
anomaly_health = fetch_anomaly_health()
anomaly_stats_data = fetch_anomaly_stats()
anomaly_results = fetch_anomaly_results(500)
anomaly_anomalies = fetch_anomaly_by_type("", 200)

# ── Parse GeoJSON → DataFrame ──
def parse_geojson_to_df(gj):
    rows = []
    if gj and "features" in gj:
        for f in gj["features"]:
            p = f.get("properties", {})
            coord = f.get("geometry", {}).get("coordinates", [None, None])
            dest_code = str(p.get("destination", "") or "")
            dest_info = get_airport_info(dest_code) or {}
            cs = str(p.get("callsign", "") or "")

            # Priority 1: route dari API (langsung dari ES, paling akurat)
            api_route = str(p.get("route", "") or "")
            origin_code = ""
            if api_route and "_" in api_route:
                origin_code = api_route.split("_")[0]

            # Priority 2: callsign lookup CSV
            if not origin_code:
                route_info = get_callsign_info(cs) or {}
                origin_code = route_info.get("origin", "")

            # Priority 3: estimasi dari heading + destination
            if not origin_code:
                est = estimate_origin_from_heading(
                    coord[1] if len(coord) > 1 else None,
                    coord[0] if len(coord) > 0 else None,
                    p.get("heading"),
                    dest_code=dest_code,
                )
                if est:
                    origin_code = est
            origin_info = get_airport_info(origin_code) or {}

            prediction_method = str(p.get("prediction_method", "") or "")
            method_label = METHOD_LABELS.get(prediction_method, prediction_method or "Unknown")

            status = str(p.get("status", "") or "")
            landed = status == "landed"
            alt = p.get("altitude", 0) or 0
            spd = p.get("speed_kmh", 0) or 0
            eta_min = p.get("eta_minutes")
            dist = p.get("distance_km_to_dest", 0) or 0

            # Estimate phase from altitude
            if alt < 1000:
                phase = "Approach"
            elif alt < 10000:
                phase = "Descent"
            elif alt < 25000:
                phase = "Climb"
            else:
                phase = "Cruise"

            # Estimate progress
            if eta_min and eta_min > 0 and spd > 100:
                total_est = dist + (eta_min / 60 * spd)
                progress = max(2, min(98, (1 - dist / total_est) * 100)) if total_est > 0 else 50
            else:
                progress = 50.0

            rows.append({
                "icao24": p.get("icao24", ""),
                "callsign": cs,
                "lat": coord[1] if len(coord) > 1 else None,
                "lon": coord[0] if len(coord) > 0 else None,
                "heading": p.get("heading", 0) or 0,
                "altitude": alt,
                "speed": spd,
                "method": method_label,
                "prediction_method": prediction_method,
                "confidence": p.get("confidence", 0) or 0,
                "eta_minutes": eta_min,
                "eta_seconds": p.get("eta_seconds"),
                "eta_method": p.get("eta_method", ""),
                "distance_km_to_dest": dist,
                "track_points": p.get("track_points"),
                "status": status,
                "landed": landed,
                "destination": dest_code,
                "dest_name": dest_info.get("name", dest_code),
                "dest_lat": dest_info.get("lat"),
                "dest_lon": dest_info.get("lon"),
                "origin": origin_code,
                "origin_name": origin_info.get("name", origin_code),
                "origin_lat": origin_info.get("lat"),
                "origin_lon": origin_info.get("lon"),
                "progress": progress,
                "phase": phase,
                "predicted_at": p.get("predicted_at"),
            })
    return pd.DataFrame(rows)

flights_df = parse_geojson_to_df(eta_gj)
anomalies_df = pd.DataFrame(anomaly_anomalies if anomaly_anomalies else [])

# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰️ SkyWatch")
    st.caption("Pengaturan koneksi backend")

    st.markdown("##### Status Layanan")
    # Cek ETA API: prioritas /eta/stats, fallback ke GeoJSON
    eta_online = eta_stats is not None or (eta_gj and len(eta_gj.get("features", [])) > 0)
    if eta_stats:
        st.success(f"✅ ETA API — {eta_stats.get('total_predictions', 0)} records")
    elif eta_online:
        st.success(f"✅ ETA API — {len(eta_gj.get('features',[]))} flights (stats unavailable)")
    else:
        st.error("❌ ETA API unreachable")
    if anomaly_health:
        st.success("✅ Anomaly API — Connected")
    else:
        st.error("❌ Anomaly API unreachable")

    st.divider()
    st.markdown("##### Filter Tampilan")
    method_filter = st.multiselect("Metode prediksi", METHODS, default=METHODS)
    conf_threshold = st.slider("Confidence minimum", 0.0, 1.0, 0.0, 0.05)
    show_trails = st.toggle("Tampilkan jejak terbang", value=True)
    show_anomaly_overlay = st.toggle("Overlay anomali di peta", value=True)

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    with st.expander("🔍 Debug", expanded=False):
        st.write(f"eta_stats: {'OK' if eta_stats else 'None'}")
        st.write(f"eta_gj: {'OK (' + str(len(eta_gj.get('features',[]))) + ' features)' if eta_gj and 'features' in eta_gj else 'None/empty'}")
        st.write(f"flights_df: {flights_df.shape if not flights_df.empty else 'EMPTY'}")
        st.write(f"anomaly_health: {'OK' if anomaly_health else 'None'}")

# ── Filter flights ──
method_map_reverse = {v: k for k, v in METHOD_LABELS.items()}
method_filter_raw = [method_map_reverse.get(m, m) for m in method_filter]
filtered_df = flights_df[
    flights_df["prediction_method"].isin(method_filter_raw) & (flights_df["confidence"] >= conf_threshold)
] if not flights_df.empty else flights_df

# ──────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%H:%M:%S, %d %b %Y")
eta_online = eta_stats is not None or (eta_gj and len(eta_gj.get("features", [])) > 0)
live_badge = "online" if (eta_online or anomaly_health) else "demo"
live_text = "LIVE" if (eta_online or anomaly_health) else "MODE DEMO"

st.markdown(
    f"""
    <div class="sw-hero">
        <div>
            <h1>🛫 SkyWatch</h1>
            <p>Dashboard Pemantauan Lalu Lintas Udara Real-Time · Prediksi ETA & Deteksi Anomali</p>
        </div>
        <div style="text-align:right;">
            <span class="sw-badge"><span class="sw-dot {live_badge}"></span> {live_text}</span>
            <p style="margin-top:8px; color:var(--sw-muted); font-size:0.8rem;">⏱ {now_str}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Gambaran Umum", "🗺️ Peta Langsung", "✈️ Peta Animasi",
    "🌐 Perjalanan Penerbangan", "📈 Riwayat & Evaluasi Model", "🚨 Deteksi Anomali",
])

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — GAMBARAN UMUM
# ════════════════════════════════════════════════════════════════════════
with tab1:
    if eta_stats:
        total_pred = eta_stats.get('total_predictions', 0)
        active = eta_stats.get('active_flights', 0)
        landed = eta_stats.get('landed_flights', 0)
        failed_pred = eta_stats.get('failed_predictions', 0)
        avg_conf = eta_stats.get('avg_confidence', 0) or 0
    else:
        total_pred = len(flights_df)
        active = int((~flights_df["landed"]).sum()) if not flights_df.empty else 0
        landed = int(flights_df["landed"].sum()) if not flights_df.empty else 0
        failed_pred = 0
        avg_conf = flights_df["confidence"].mean() if not flights_df.empty else 0

    cols = st.columns(5)
    metric_data = [
        ("Total Prediksi", f"{total_pred:,}", "📡", "+ live", "var(--sw-accent)"),
        ("Pesawat Aktif", f"{active:,}", "🛩️", "di udara", "var(--sw-green)"),
        ("Sudah Mendarat", f"{landed:,}", "🛬", "selesai", "var(--sw-muted)"),
        ("Prediksi Gagal", f"{failed_pred:,}", "⚠️", "perlu cek", "var(--sw-amber)"),
        ("Rata-rata Confidence", f"{avg_conf*100:.1f}%" if avg_conf else "N/A", "🎯", "akurasi model", "var(--sw-accent-2)"),
    ]
    for c, (label, value, icon, sub, color) in zip(cols, metric_data):
        with c:
            st.markdown(
                f"""<div class="metric-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span class="metric-label">{label}</span><span class="metric-icon">{icon}</span>
                        </div>
                        <div class="metric-value" style="color:{color}">{value}</div>
                        <div class="metric-sub" style="color:{color}">{sub}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("##### 🏆 Destinasi Terpopuler")
        if not flights_df.empty and "destination" in flights_df.columns:
            dest_counts = flights_df["destination"].value_counts().head(8).reset_index()
            dest_counts.columns = ["destinasi", "jumlah"]
            fig = px.bar(
                dest_counts, x="jumlah", y="destinasi", orientation="h",
                color="jumlah", color_continuous_scale=["#1e3a5f", "#38bdf8"],
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", showlegend=False, coloraxis_showscale=False,
                yaxis=dict(autorange="reversed", gridcolor="#233252"),
                xaxis=dict(gridcolor="#233252"), height=320, margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("##### 🧮 Distribusi Metode Prediksi")
        if not flights_df.empty and "method" in flights_df.columns:
            method_counts = flights_df["method"].value_counts().reset_index()
            method_counts.columns = ["metode", "jumlah"]
            fig2 = px.pie(
                method_counts, names="metode", values="jumlah", hole=0.55,
                color="metode",
                color_discrete_map={"Route Lookup": "#38bdf8", "ML Classifier": "#34d399", "Heading Score": "#fbbf24"},
            )
        fig2.update_traces(textfont_color="#0b1220", textfont_size=13)
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=320, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### ✈️ Daftar Pesawat Terkini")
    if not flights_df.empty:
        show_df = flights_df.copy()
        show_df["status_label"] = show_df["landed"].map({True: "🛬 Mendarat", False: "🛫 Di Udara"})
        show_df["asal"] = show_df.apply(lambda r: r.get("origin") if r.get("origin") else "-", axis=1)
        display_cols = [c for c in ["callsign", "icao24", "asal", "destination", "altitude",
                                     "speed", "method", "confidence", "status_label"] if c in show_df.columns]
        st.dataframe(
            show_df[display_cols].rename(columns={
                "callsign": "Callsign", "icao24": "ICAO24", "asal": "Asal", "destination": "Tujuan",
                "altitude": "Altitude (ft)", "speed": "Kecepatan (km/h)", "method": "Metode",
                "confidence": "Confidence", "status_label": "Status",
            }),
            use_container_width=True, height=320,
            column_config={"Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f")},
        )

# ════════════════════════════════════════════════════════════════════════
# TAB 2 — PETA LANGSUNG (Folium)
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("##### 🗺️ Peta Langsung — Posisi Real-Time Pesawat")
    map_c1, map_c2 = st.columns([3.3, 1])

    with map_c2:
        st.markdown('<div class="sw-panel">', unsafe_allow_html=True)
        st.markdown("**Filter Peta**")
        st.caption("Gunakan filter di sidebar untuk menyesuaikan metode & confidence.")
        st.metric("Pesawat ditampilkan", len(filtered_df))
        sel_callsign_options = ["—"]
        if not filtered_df.empty:
            sel_callsign_options += sorted(filtered_df["callsign"].astype(str).unique().tolist())
        sel_callsign = st.selectbox("Pilih pesawat", sel_callsign_options)
        st.markdown("</div>", unsafe_allow_html=True)
        st.session_state["selected_flight"] = sel_callsign

    method_colors_map = {"Route Lookup": "#38bdf8", "ML Classifier": "#34d399", "Heading Score": "#fbbf24", "heading_scoring": "#fbbf24"}

    with map_c1:
        m = folium.Map(location=[50, 10], zoom_start=4, tiles="CartoDB dark_matter", control_scale=True)

        for _, f in filtered_df.iterrows():
            color = method_colors_map.get(f.get("method"), "#38bdf8")
            if pd.isna(f.get("lat")) or pd.isna(f.get("lon")):
                continue
            is_selected = str(f.get("callsign")) == sel_callsign
            radius = 9 if is_selected else 6
            origin_display = f.get("origin", "") or "-"
            popup_html = f"""
                <b>{f.get('callsign','-')}</b> ({f.get('icao24','-')})<br>
                Asal → Tujuan: {origin_display} → {f.get('destination','-')}<br>
                Altitude: {f.get('altitude',0):.0f} ft<br>
                Kecepatan: {f.get('speed',0):.0f} km/h<br>
                Metode: {f.get('method','-')}<br>
                Confidence: {round(float(f.get('confidence',0))*100,1)}%
            """
            folium.CircleMarker(
                location=[f["lat"], f["lon"]], radius=radius, color=color, fill=True,
                fill_color=color, fill_opacity=0.85, weight=2,
                popup=folium.Popup(popup_html, max_width=260), tooltip=f.get("callsign", ""),
            ).add_to(m)

            origin_lat = f.get("origin_lat")
            origin_lon = f.get("origin_lon")
            if show_trails and origin_lat is not None and origin_lon is not None and not pd.isna(origin_lat):
                folium.PolyLine(
                    locations=[[origin_lat, origin_lon], [f["lat"], f["lon"]]],
                    color=color, weight=1.4, opacity=0.35, dash_array="4,4",
                ).add_to(m)

            dest_lat = f.get("dest_lat")
            dest_lon = f.get("dest_lon")
            if dest_lat is not None and dest_lon is not None and not pd.isna(dest_lat):
                folium.PolyLine(
                    locations=[[f["lat"], f["lon"]], [dest_lat, dest_lon]],
                    color=color, weight=1, opacity=0.2, dash_array="2,6",
                ).add_to(m)

        if show_anomaly_overlay and not anomalies_df.empty:
            anomaly_icaos = set(anomalies_df["icao24"].astype(str)) if "icao24" in anomalies_df.columns else set()
            for _, f in filtered_df.iterrows():
                if str(f.get("icao24")) in anomaly_icaos:
                    folium.CircleMarker(
                        location=[f["lat"], f["lon"]], radius=16, color="#f87171",
                        fill=False, weight=2, dash_array="3,4",
                        tooltip="⚠️ Anomali terdeteksi",
                    ).add_to(m)

        st_folium(m, use_container_width=True, height=560, returned_objects=[])

    st.caption("💡 Klik ikon pesawat untuk melihat detail. Pilih pesawat lewat dropdown di samping untuk menyorot di semua tab.")

# ════════════════════════════════════════════════════════════════════════
# TAB 3 — PETA ANIMASI (Leaflet via HTML embed dengan GeoJSON real-time)
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("##### ✈️ Peta Animasi — Pergerakan Pesawat Halus (Leaflet)")
    st.caption("Interpolasi posisi 5 detik · Refresh GeoJSON tiap 10 detik · Data real dari API.")

    gj_data = eta_gj if eta_gj else {"type": "FeatureCollection", "features": []}
    initial_json = json.dumps(gj_data)
    airport_data = {}
    if gj_data and "features" in gj_data:
        for f in gj_data["features"]:
            dest = f["properties"].get("destination", "")
            if dest and dest not in airport_data:
                info = get_airport_info(dest)
                if info and info.get("lat") and info.get("lon"):
                    airport_data[dest] = {"lat": info["lat"], "lon": info["lon"], "name": info.get("name", dest)}
    airport_json = json.dumps(airport_data)
    mcolors = json.dumps(METHOD_COLORS)
    mlabels = json.dumps(METHOD_LABELS)

    leaflet_html = f"""<div id="sw-anim-map" style="width:100%;height:600px;border-radius:16px;overflow:hidden;border:1px solid #233252;font-family:Inter,sans-serif;">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<div id="alegend" style="position:absolute;bottom:24px;left:12px;z-index:1000;background:#121a2b;border:1px solid #233252;border-radius:12px;padding:10px 14px;font-size:12px;color:#8aa0c4;box-shadow:0 3px 10px rgba(0,0,0,0.3);">
  <div style="font-weight:700;margin-bottom:6px;font-size:13px;color:#e2e8f0;">✈ Legend</div>
  <div><span style="display:inline-block;width:12px;height:3px;border-radius:2px;background:#38bdf8;margin-right:6px;"></span>Route Lookup</div>
  <div><span style="display:inline-block;width:12px;height:3px;border-radius:2px;background:#34d399;margin-right:6px;"></span>ML Classifier</div>
  <div><span style="display:inline-block;width:12px;height:3px;border-radius:2px;background:#fbbf24;margin-right:6px;"></span>Heading Score</div>
  <div style="margin-top:6px;color:#8aa0c4;font-size:11px;"><span id="acount">0</span> tracked</div>
</div>
<div id="astats" style="position:absolute;top:12px;right:12px;z-index:1000;background:#121a2b;border:1px solid #233252;border-radius:12px;padding:8px 12px;font-size:11px;color:#8aa0c4;box-shadow:0 3px 10px rgba(0,0,0,0.3);text-align:right;">
  <div style="font-weight:700;color:#e2e8f0;font-size:12px;">LIVE</div>
  <div id="astime" style="font-family:JetBrains Mono,monospace;font-size:12px;color:#38bdf8;">--:--:--</div>
</div>
<style>
  .aircraft-icon {{ background:none !important; border:none !important; }}
  .leaflet-popup-content-wrapper {{ border-radius:12px; background:#121a2b; color:#e2e8f0; border:1px solid #233252; }}
  .leaflet-popup-content {{ margin:12px 14px; font-family:Inter,sans-serif; }}
  .leaflet-container {{ background: #0b1220; }}
</style>
<script>
(function(){{
  var K='__am2';if(window[K]){{window[K].u({initial_json});return;}}
  var S=window[K]={{m:null,d:new Map()}},DATA={initial_json},COLORS={mcolors},LABELS={mlabels},APTS={airport_json};
  S.m=L.map('sw-anim-map',{{center:[50,10],zoom:4,zoomControl:true,attributionControl:false}});
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{attribution:'&copy; OSM &copy; CARTO',maxZoom:19}}).addTo(S.m);
  function ic(h,c){{return L.divIcon({{html:'<div style="transform:rotate('+h+'deg);font-size:24px;line-height:1;filter:drop-shadow(0 0 5px '+c+');color:'+c+';">&#9992;</div>',iconSize:[24,24],iconAnchor:[12,12],className:'aircraft-icon'}});}}
  function pop(p){{
    var e=p.e!=null?Math.round(p.e)+' min':'N/A',co=p.co!=null?Math.round(p.co*100)+'%':'N/A',a=p.a?Math.round(p.a)+' ft':'--',s=p.s?Math.round(p.s)+' km/h':'--',h=p.h!=null?Math.round(p.h)+'\u00B0':'--';
    return '<div style="padding:12px;min-width:220px;">'+
      '<div style="font-size:16px;font-weight:800;">'+(p.cs||'N/A')+'</div>'+
      '<div style="font-size:11px;color:#8aa0c4;">ICAO24: '+(p.id||'')+'</div>'+
      '<div style="margin-top:6px;font-size:12px;">'+
        '<div>Destination: <b style="color:#38bdf8;">'+(p.d||'?')+'</b></div>'+
        '<div>ETA: <b style="color:#34d399;">'+e+'</b></div>'+
        '<div>Method: <b style="color:'+p.c+';">'+(LABELS[p.m]||p.m||'?')+'</b></div>'+
        '<div>Confidence: <b style="color:#34d399;">'+co+'</b></div>'+
      '</div>'+
      '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:6px;font-size:11px;">'+
        '<div style="text-align:center;padding:4px;background:#16213a;border-radius:6px;">ALT<br><b>'+a+'</b></div>'+
        '<div style="text-align:center;padding:4px;background:#16213a;border-radius:6px;">SPD<br><b>'+s+'</b></div>'+
        '<div style="text-align:center;padding:4px;background:#16213a;border-radius:6px;">HDG<br><b>'+h+'</b></div>'+
      '</div>'+
    '</div>';
  }}
  function proc(arr){{
    if(!arr)return;var now=Date.now(),ids=new Set();
    for(var i=0;i<arr.length;i++)try{{
      var f=arr[i];if(!f||!f.geometry||!f.properties)continue;
      var c=f.geometry.coordinates;if(!c||c.length<2)continue;
      var lon=c[0],lat=c[1],p=f.properties,id=p.icao24;if(!id)continue;
      ids.add(id);
      if(S.d.has(id)){{
        var a=S.d.get(id),ch=Math.abs(a.tlat-lat)>0.001||Math.abs(a.tlon-lon)>0.001;
        a.plat=a.clat;a.plon=a.clon;a.tlat=lat;a.tlon=lon;a.h=p.heading||a.h||0;
        if(ch)a.as=performance.now();a.cs=p.callsign||a.cs;a.d=p.destination||a.d;a.a=p.altitude||a.a;
        a.s=p.speed_kmh||a.s;a.co=p.confidence??a.co;a.e=p.eta_minutes??a.e;a.m=p.prediction_method||a.m;a.ts=now;
        a.mk.setPopupContent(pop({{id:id,cs:a.cs,d:a.d,a:a.a,s:a.s,co:a.co,e:a.e,m:a.m,h:a.h,c:a.c,lat:lat,lon:lon}}));
      }}else{{
        var c2=COLORS[p.prediction_method]||'#8aa0c4';
        var mk=L.marker([lat,lon],{{icon:ic(p.heading||0,c2)}}).addTo(S.m);
        var tl=L.polyline([[lat,lon]],{{color:c2,weight:2,opacity:0.5,dashArray:'5,8'}}).addTo(S.m);
        mk.bindPopup(pop({{id:id,cs:p.callsign,d:p.destination,a:p.altitude,s:p.speed_kmh,co:p.confidence,e:p.eta_minutes,m:p.prediction_method,h:p.heading||0,c:c2,lat:lat,lon:lon}}));
        var dl=null,dm=null;
        if(APTS[p.destination]){{var da=APTS[p.destination];dl=L.polyline([[lat,lon],[da.lat,da.lon]],{{color:c2,weight:1.5,opacity:0.35,dashArray:'4,12'}}).addTo(S.m);dm=L.circleMarker([da.lat,da.lon],{{radius:5,color:c2,fillColor:c2,fillOpacity:0.4,weight:1.5}}).addTo(S.m);}}
        S.d.set(id,{{mk:mk,tl:tl,dl:dl,dm:dm,plat:lat,plon:lon,tlat:lat,tlon:lon,clat:lat,clon:lon,h:p.heading||0,cs:p.callsign,d:p.destination,a:p.altitude,s:p.speed_kmh,co:p.confidence,e:p.eta_minutes,m:p.prediction_method,c:c2,thist:[[lat,lon]],ts:now,as:performance.now(),id:id}});
      }}
    }}catch(e){{}}
    S.d.forEach(function(a,id){{if(!ids.has(id)&&now-a.ts>30000){{try{{S.m.removeLayer(a.mk);S.m.removeLayer(a.tl);if(a.dl)S.m.removeLayer(a.dl);if(a.dm)S.m.removeLayer(a.dm);}}catch(x){{}}S.d.delete(id);}}}});
    var el=document.getElementById('acount');if(el)el.textContent=S.d.size;
  }}
  function anim(t){{
    S.d.forEach(function(a){{
      var el=t-a.as,dur=5000,f=Math.min(1,el/dur),ease=f<0.5?2*f*f:1-Math.pow(-2*f+2,2)/2;
      a.clat=a.plat+(a.tlat-a.plat)*ease;a.clon=a.plon+(a.tlon-a.plon)*ease;
      a.mk.setLatLng([a.clat,a.clon]);a.thist.push([a.clat,a.clon]);if(a.thist.length>25)a.thist.shift();
      a.tl.setLatLngs(a.thist);
      if(a.dl&&APTS[a.d]){{var da=APTS[a.d];a.dl.setLatLngs([[a.clat,a.clon],[da.lat,da.lon]]);}}
      a.mk.setIcon(ic(a.h||0,a.c));if(f>=1){{a.plat=a.tlat;a.plon=a.tlon;}}
    }});requestAnimationFrame(anim);
  }}
  S.u=function(d){{if(d&&d.features)proc(d.features);}};
  if(DATA&&DATA.features)proc(DATA.features);
  requestAnimationFrame(anim);
  setInterval(function(){{fetch('{ETA_API_HOST}/eta/predictions.geojson?limit=300').then(function(r){{return r.json();}}).then(function(d){{if(d&&d.features)proc(d.features);}}).catch(function(){{}});}},10000);
  setInterval(function(){{try{{S.m.invalidateSize();}}catch(e){{}}}},2000);
  setInterval(function(){{var d=new Date(),el=document.getElementById('astime');if(el)el.textContent=d.toTimeString().slice(0,8);}},1000);
}})();
</script>
</div>"""
    components.html(leaflet_html, height=610)

# ════════════════════════════════════════════════════════════════════════
# TAB 4 — PERJALANAN PENERBANGAN
# ════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("##### 🌐 Detail Perjalanan Penerbangan")
    if not filtered_df.empty:
        callsigns = sorted(filtered_df["callsign"].astype(str).unique().tolist())
        sel_callsign = st.session_state.get("selected_flight", "—") if "selected_flight" in st.session_state else "—"
        default_idx = callsigns.index(sel_callsign) if sel_callsign in callsigns else 0
        chosen = st.selectbox("Pilih penerbangan", callsigns, index=default_idx, key="journey_select")
        f = filtered_df[filtered_df["callsign"].astype(str) == chosen].iloc[0]

        phase = f.get("phase", "Cruise")
        phase_colors = {
            "Takeoff": "#fbbf24", "Climb": "#38bdf8", "Cruise": "#34d399",
            "Descent": "#a78bfa", "Approach": "#f87171",
        }
        pc = phase_colors.get(phase, "#38bdf8")

        origin_display = f.get("origin", "") or "-"
        dest_display = f.get("destination", "") or "-"
        origin_name = f.get("origin_name", "") or "-"
        dest_name = f.get("dest_name", "") or "-"

        st.markdown(
            f"""<div class="sw-panel" style="margin-bottom:16px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h3 style="margin:0;">{f.get('callsign')} · {origin_display} → {dest_display}</h3>
                        <p style="color:var(--sw-muted); margin:2px 0 0 0;">{origin_name} menuju {dest_name}</p>
                    </div>
                    <span class="phase-badge" style="color:{pc}; border-color:{pc};">{phase.upper()}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # progress arc path
        prog = float(f.get("progress", 50))
        o_lat = f.get("origin_lat")
        o_lon = f.get("origin_lon")
        d_lat = f.get("dest_lat")
        d_lon = f.get("dest_lon")

        if o_lat is not None and o_lon is not None and d_lat is not None and d_lon is not None and not pd.isna(o_lat):
            n_pts = 60
            t = np.linspace(0, 1, n_pts)
            mid_lat = (o_lat + d_lat) / 2 + 3
            mid_lon = (o_lon + d_lon) / 2
            curve_lat = (1-t)**2*o_lat + 2*(1-t)*t*mid_lat + t**2*d_lat
            curve_lon = (1-t)**2*o_lon + 2*(1-t)*t*mid_lon + t**2*d_lon
            split = int(prog/100*n_pts)

            fig_path = go.Figure()
            fig_path.add_trace(go.Scattergeo(
                lon=curve_lon, lat=curve_lat, mode="lines",
                line=dict(width=2, color="#233252", dash="dot"), name="Rute"
            ))
            fig_path.add_trace(go.Scattergeo(
                lon=curve_lon[:split+1], lat=curve_lat[:split+1], mode="lines",
                line=dict(width=3.5, color="#38bdf8"), name="Progres"
            ))
            fig_path.add_trace(go.Scattergeo(
                lon=[o_lon, d_lon], lat=[o_lat, d_lat], mode="markers+text",
                marker=dict(size=10, color=["#34d399", "#f87171"]),
                text=[origin_display, dest_display], textposition="top center",
                textfont=dict(color="#e2e8f0"), name="Bandara"
            ))
            if split < n_pts:
                fig_path.add_trace(go.Scattergeo(
                    lon=[curve_lon[split]], lat=[curve_lat[split]], mode="markers",
                    marker=dict(size=14, color="#fbbf24", symbol="triangle-right"), name="Pesawat"
                ))
            fig_path.update_geos(
                projection_type="natural earth", showland=True, landcolor="#16213a",
                showocean=True, oceancolor="#0b1220", showcountries=True, countrycolor="#233252",
                bgcolor="rgba(0,0,0,0)",
            )
            fig_path.update_layout(
                height=420, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", showlegend=False,
            )
            st.plotly_chart(fig_path, use_container_width=True)
        else:
            st.info("Data rute tidak tersedia (origin/destination bandara tidak diketahui).")

        eta_min_val = f.get("eta_minutes")
        eta_display = f"{eta_min_val:.0f}" if eta_min_val is not None else "-"
        dist = f.get("distance_km_to_dest", 0) or 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⏱ ETA", f"{eta_display} menit")
        m2.metric("📏 Jarak Tersisa", f"{dist:,.0f} km" if dist else "-")
        m3.metric("🎯 Confidence", f"{float(f.get('confidence',0))*100:.1f}%")
        m4.metric("📊 Progres", f"{prog:.1f}%")

        g1, g2 = st.columns(2)
        with g1:
            alt_val = float(f.get("altitude", 30000))
            fig_alt = go.Figure(go.Indicator(
                mode="gauge+number", value=alt_val,
                title={"text": "Altitude (ft)", "font": {"color": "#e2e8f0"}},
                number={"font": {"color": "#38bdf8"}},
                gauge={
                    "axis": {"range": [0, 42000], "tickcolor": "#8aa0c4"},
                    "bar": {"color": "#38bdf8"},
                    "bgcolor": "#16213a", "borderwidth": 1, "bordercolor": "#233252",
                    "steps": [
                        {"range": [0, 10000], "color": "#1e2a44"},
                        {"range": [10000, 30000], "color": "#1c3a55"},
                        {"range": [30000, 42000], "color": "#16475f"},
                    ],
                },
            ))
            fig_alt.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig_alt, use_container_width=True)
        with g2:
            spd_val = float(f.get("speed", 800))
            fig_spd = go.Figure(go.Indicator(
                mode="gauge+number", value=spd_val,
                title={"text": "Kecepatan (km/h)", "font": {"color": "#e2e8f0"}},
                number={"font": {"color": "#34d399"}},
                gauge={
                    "axis": {"range": [0, 1100], "tickcolor": "#8aa0c4"},
                    "bar": {"color": "#34d399"},
                    "bgcolor": "#16213a", "borderwidth": 1, "bordercolor": "#233252",
                    "steps": [
                        {"range": [0, 400], "color": "#1e2a44"},
                        {"range": [400, 800], "color": "#1c3a55"},
                        {"range": [800, 1100], "color": "#16475f"},
                    ],
                },
            ))
            fig_spd.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig_spd, use_container_width=True)
    else:
        st.info("Tidak ada penerbangan yang sesuai filter saat ini.")

# ════════════════════════════════════════════════════════════════════════
# TAB 5 — RIWAYAT & EVALUASI MODEL
# ════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("##### 📈 Riwayat Prediksi & Evaluasi Model")
    if not flights_df.empty:
        c1, c2 = st.columns([1, 3])
        with c1:
            callsign_list = sorted(flights_df["callsign"].astype(str).unique().tolist())
            hist_flight = st.selectbox("Pilih pesawat", callsign_list, key="hist_select")
            time_range = st.select_slider("Rentang waktu", options=["1 Jam", "3 Jam", "6 Jam", "12 Jam", "24 Jam"], value="3 Jam")

        hours_map = {"1 Jam": 1, "3 Jam": 3, "6 Jam": 6, "12 Jam": 12, "24 Jam": 24}
        hist_hours = hours_map.get(time_range, 3)
        hist_data = fetch_eta_history(hist_flight.split(" (")[0] if "(" in hist_flight else hist_flight, hours=hist_hours)

        with c2:
            if hist_data and "history" in hist_data and len(hist_data["history"]) > 0:
                hx = hist_data["history"]
                times_h = []
                eta_vals = []
                conf_vals = []
                dist_vals = []
                for h in hx:
                    ts = h.get("recorded_at") or h.get("predicted_at")
                    if ts:
                        try:
                            t_parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            times_h.append(t_parsed)
                            eta_vals.append(h.get("eta_minutes"))
                            conf_vals.append(h.get("confidence"))
                            dist_vals.append(h.get("distance_km_to_dest"))
                        except Exception:
                            pass

                if len(times_h) >= 2:
                    mm1, mm2, mm3 = st.columns(3)
                    mm1.metric("Total Data Points", len(times_h))
                    mm2.metric("ETA Terakhir", f"{eta_vals[-1]:.1f} min" if eta_vals[-1] else "N/A")
                    mm3.metric("Confidence Terakhir", f"{conf_vals[-1]*100:.1f}%" if conf_vals[-1] else "N/A")

                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(x=times_h, y=eta_vals, name="ETA (min)", line=dict(color="#38bdf8", width=2)))
                    fig_hist.update_layout(
                        title=f"Riwayat ETA — {hist_flight}", height=320,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                        xaxis=dict(gridcolor="#233252"), yaxis=dict(gridcolor="#233252", title="ETA (menit)"),
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    g1, g2 = st.columns(2)
                    with g1:
                        fig_conf = go.Figure()
                        fig_conf.add_trace(go.Scatter(x=times_h, y=conf_vals, name="Confidence", line=dict(color="#fbbf24", width=2)))
                        fig_conf.update_layout(
                            title="Tren Confidence", height=280,
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                            xaxis=dict(gridcolor="#233252"), yaxis=dict(gridcolor="#233252", title="Confidence", range=[0, 1]),
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)
                    with g2:
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Scatter(x=times_h, y=dist_vals, name="Jarak", line=dict(color="#38bdf8", width=2), fill="tozeroy"))
                        fig_dist.update_layout(
                            title="Jarak ke Destinasi", height=280,
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                            xaxis=dict(gridcolor="#233252"), yaxis=dict(gridcolor="#233252", title="Jarak (km)"),
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                else:
                    st.info("Data histori belum mencukupi untuk ditampilkan.")
            else:
                st.info("Tidak ada data histori untuk pesawat ini.")
    else:
        st.info("Tidak ada data penerbangan.")

# ════════════════════════════════════════════════════════════════════════
# TAB 6 — DETEKSI ANOMALI
# ════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("##### 🚨 Deteksi Anomali — Monitoring Real-Time")

    if anomaly_stats_data:
        total_anom = anomaly_stats_data.get("anomalies", anomaly_stats_data.get("total", 0))
        normal_count = anomaly_stats_data.get("normal", 0)
        anom_rate = anomaly_stats_data.get("rate", 0)
    else:
        total_anom = len(anomalies_df) if not anomalies_df.empty else 0
        anom_rate = 0

    a1, a2, a3, a4 = st.columns(4)
    for col, (label, val, icon, color) in zip(
        [a1, a2, a3, a4],
        [
            ("Total Anomali", total_anom, "🚨", "var(--sw-red)"),
            ("Terdeteksi", len(anomalies_df) if not anomalies_df.empty else 0, "🔍", "var(--sw-amber)"),
            ("Anomaly Rate", f"{anom_rate:.1%}" if anom_rate else "N/A", "📊", "var(--sw-accent)"),
            ("Status API", "Connected" if anomaly_health else "Offline", "📡", "var(--sw-green)" if anomaly_health else "var(--sw-red)"),
        ],
    ):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color};">{val} {icon}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    b1, b2 = st.columns([2, 1.2])

    with b1:
        st.markdown("**Distribusi Jenis Anomali**")
        if not anomalies_df.empty and "attack_type" in anomalies_df.columns:
            type_counts = anomalies_df["attack_type"].value_counts().reset_index()
            type_counts.columns = ["jenis", "jumlah"]
            fig_types = px.bar(type_counts, x="jenis", y="jumlah", color="jumlah",
                                color_continuous_scale=["#3b1f1f", "#f87171"])
            fig_types.update_layout(
                height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                xaxis=dict(gridcolor="#233252", title=""), yaxis=dict(gridcolor="#233252", title="Jumlah"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_types, use_container_width=True)
        else:
            st.info("Belum ada data anomali dari API.")

    with b2:
        st.markdown("**📡 Feed Anomali Terbaru**")
        st.markdown('<div class="sw-panel" style="max-height:660px; overflow-y:auto;">', unsafe_allow_html=True)
        if not anomalies_df.empty:
            for _, row in anomalies_df.head(20).iterrows():
                attack_type = row.get("attack_type", row.get("jenis", "-"))
                callsign = row.get("callsign", "-")
                icao = row.get("icao24", "-")
                score = row.get("combined_score", row.get("reconstruction_error", 0))
                score_pct = f"{float(score)*100:.0f}%" if score else "-"
                st.markdown(
                    f"""<div class="flight-row">
                        <div>
                            <b>{callsign}</b> <span style="color:var(--sw-muted); font-size:0.78rem;">{icao}</span><br>
                            <span style="font-size:0.82rem; color:var(--sw-muted);">{attack_type} · Score: {score_pct}</span>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Tidak ada anomali terdeteksi saat ini. ✅")
        st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# FOOTER & AUTO-REFRESH
# ──────────────────────────────────────────────────────────────────────────
st.write("")
st.markdown(
    """<div style="text-align:center; color:var(--sw-muted); font-size:0.78rem; padding:20px 0;">
    SkyWatch · ETA API &amp; Anomaly API · Data Real-time · Auto-refresh setiap {} detik
    </div>""".format(AUTO_REFRESH_SEC),
    unsafe_allow_html=True,
)

st_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")
