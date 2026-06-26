# ETA Flight Prediction API

Base URL: `http://100.94.21.31:8002`

## Prasyarat

Laptop harus terhubung ke **Tailscale** (satu network dengan laptop server).

---

## Endpoints

### 1. Cek Koneksi

```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "es_connected": true,
  "total_predictions": 3500
}
```

**Cek:**
```bash
curl http://100.94.21.31:8002/health
```

---

### 2. Data Penerbangan untuk Map

```
GET /eta/predictions.geojson
```

**Response:** Format GeoJSON — langsung bisa diplot ke Mapbox / Leaflet.

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [6.95, 46.31]
      },
      "properties": {
        "icao24": "a2cf48",
        "callsign": "GJE622",
        "destination": "LFPG",
        "eta_minutes": 272.1,
        "altitude": 14295.0,
        "heading": 329.0,
        "speed_kmh": 847.0,
        "confidence": 0.74,
        "status": "ok"
      }
    }
  ]
}
```

**Filter yang tersedia:**
```
?destination=EGLL          ← filter bandara tujuan
?status=ok                 ← filter status prediksi
?method=ml_classifier      ← filter metode prediksi
?limit=200                 ← jumlah data (max 1000)
?offset=0                  ← halaman
```

**Contoh:**
```bash
curl "http://100.94.21.31:8002/eta/predictions.geojson?destination=EGLL&limit=50"
```

---

### 3. Semua Prediksi (JSON)

```
GET /eta/predictions
```

**Parameter (optional):**
| Parameter | Tipe | Default | Contoh |
|-----------|------|---------|--------|
| `destination` | string | - | `EGLL` |
| `status` | string | - | `ok`, `failed`, `no_data` |
| `method` | string | - | `ml_classifier`, `callsign`, `heading_scoring` |
| `limit` | int | 20 | `100` |
| `offset` | int | 0 | `0` |

**Contoh:**
```bash
curl "http://100.94.21.31:8002/eta/predictions?limit=5"
```

---

### 4. Detail Satu Pesawat

```
GET /eta/predictions/{icao24}
```

**Contoh:**
```bash
curl http://100.94.21.31:8002/eta/predictions/a2cf48
```

**Response:**
```json
{
  "icao24": "a2cf48",
  "callsign": "GJE622",
  "destination": "LFPG",
  "prediction_method": "ml_classifier",
  "confidence": 0.74,
  "eta_minutes": 272.1,
  "current_position": {
    "lat": 46.31,
    "lon": 6.95,
    "altitude": 14295.0,
    "heading": 329.0,
    "speed_kmh": 847.0
  },
  "status": "ok"
}
```

---

### 5. Prediksi Manual

```
POST /eta/predict
```

**Body (JSON):**
```json
{
  "icao24": "a2cf48"
}
```

**Contoh:**
```bash
curl -X POST http://100.94.21.31:8002/eta/predict \
  -H "Content-Type: application/json" \
  -d '{"icao24":"a2cf48"}'
```

---

### 6. Statistik Dashboard

```
GET /eta/stats
```

**Response:**
```json
{
  "total_predictions": 3500,
  "active_flights": 3480,
  "top_destinations": [
    {"destination": "EGLL", "count": 320},
    {"destination": "LFPG", "count": 280}
  ],
  "avg_confidence": 0.76
}
```

**Contoh:**
```bash
curl http://100.94.21.31:8002/eta/stats
```

---

## Integrasi ke Map

### Mapbox GL JS

```js
async function loadFlights() {
  const res = await fetch('http://100.94.21.31:8002/eta/predictions.geojson');
  const data = await res.json();

  map.addSource('flights', {
    type: 'geojson',
    data: data
  });

  map.addLayer({
    id: 'flights',
    type: 'circle',
    source: 'flights',
    paint: {
      'circle-radius': 5,
      'circle-color': '#ff4444',
      'circle-opacity': 0.8
    }
  });
}
// Refresh tiap 30 detik
setInterval(loadFlights, 30000);
```

### Leaflet

```js
async function loadFlights() {
  const res = await fetch('http://100.94.21.31:8002/eta/predictions.geojson');
  const data = await res.json();

  L.geoJSON(data, {
    pointToLayer: function(feature, latlng) {
      return L.circleMarker(latlng, {
        radius: 5,
        color: '#ff4444',
        fillOpacity: 0.8
      });
    }
  }).addTo(map);
}
// Refresh tiap 30 detik
setInterval(loadFlights, 30000);
```

---

## Catatan

- Data diupdate setiap **30 detik** (scheduler).
- Posisi pesawat (lat/lon) adalah posisi **real-time terakhir** dari OpenSky.
- `eta_minutes` adalah perkiraan waktu tersisa sampai mendarat di bandara tujuan.
- `confidence` (0.0 - 1.0) adalah skor kepercayaan prediksi.
- Kalau endpoint `predictions.geojson` lambat, tambah `?limit=100` untuk batasi data.
