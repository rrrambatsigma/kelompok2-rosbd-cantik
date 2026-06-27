import requests

r = requests.get('http://100.99.130.69:9200/flights/_search', json={
    'size': 5,
    'query': {'term': {'icao24.keyword': '344045'}},
    'sort': [{'timestamp': {'order': 'asc'}}],
    '_source': ['icao24', 'latitude', 'longitude', 'velocity', 'baro_altitude', 'true_track', 'timestamp']
}, timeout=15)

for h in r.json()['hits']['hits']:
    s = h['_source']
    lat = s.get('latitude')
    lon = s.get('longitude')
    vel = s.get('velocity')
    alt = s.get('baro_altitude')
    track = s.get('true_track')
    ts = s.get('timestamp')
    print(f'lat={lat} lon={lon} vel={vel} alt={alt} track={track} ts={ts}')
    print(f'  vel check: {vel is not None}', end='')
    if vel is not None:
        print(f' 0<={vel}<=500: {0 <= vel <= 500}', end='')
    print(f'  alt check: {alt is not None}', end='')
    if alt is not None:
        print(f' -1000<={alt}<=45000: {-1000 <= alt <= 45000}', end='')
    print()

# Also check aggregate statistics
r2 = requests.get('http://100.99.130.69:9200/flights/_search', json={
    'size': 0,
    'query': {'term': {'icao24.keyword': '344045'}},
    'aggs': {
        'null_lat': {'missing': {'field': 'latitude'}},
        'null_lon': {'missing': {'field': 'longitude'}},
        'null_vel': {'missing': {'field': 'velocity'}},
        'null_alt': {'missing': {'field': 'baro_altitude'}},
        'null_track': {'missing': {'field': 'true_track'}},
        'vel_stats': {'stats': {'field': 'velocity'}},
        'alt_stats': {'stats': {'field': 'baro_altitude'}},
        'track_stats': {'stats': {'field': 'true_track'}},
    }
}, timeout=15)

aggs = r2.json()['aggregations']
print('\nNull counts:')
print(f'  lat: {aggs["null_lat"]["doc_count"]}')
print(f'  lon: {aggs["null_lon"]["doc_count"]}')
print(f'  vel: {aggs["null_vel"]["doc_count"]}')
print(f'  alt: {aggs["null_alt"]["doc_count"]}')
print(f'  track: {aggs["null_track"]["doc_count"]}')
print(f'\nVelocity stats: {aggs["vel_stats"]}')
print(f'Altitude stats: {aggs["alt_stats"]}')
print(f'Track stats: {aggs["track_stats"]}')
