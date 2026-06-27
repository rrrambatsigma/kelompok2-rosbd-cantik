import requests

# Cari flight dengan velocity > 50 AND altitude NOT null
r = requests.get('http://100.99.130.69:9200/flights/_search', json={
    'size': 0,
    'query': {
        'bool': {
            'filter': [
                {'range': {'velocity': {'gt': 50}}},
                {'exists': {'field': 'baro_altitude'}},
                {'range': {'baro_altitude': {'gt': 1000}}}
            ]
        }
    },
    'aggs': {
        'top_flights': {
            'terms': {'field': 'icao24.keyword', 'size': 10, 'min_doc_count': 100}
        }
    }
}, timeout=15)

buckets = r.json()['aggregations']['top_flights']['buckets']
print('Top flying flights:')
for b in buckets:
    print(f'  icao24={b["key"]}, records={b["doc_count"]}')
