import requests

# Try with .keyword subfield
r = requests.get('http://100.99.130.69:9200/flights/_search', json={
    'size': 0,
    'aggs': {
        'top_flights': {
            'terms': {'field': 'icao24.keyword', 'size': 10, 'min_doc_count': 20}
        }
    }
}, timeout=15)

data = r.json()
if 'aggregations' in data:
    buckets = data['aggregations']['top_flights']['buckets']
    print('Top flights with most records:')
    for b in buckets:
        print(f'  icao24={b["key"]}, records={b["doc_count"]}')
elif 'error' in data:
    print('ES Error:', data['error']['root_cause'][0]['reason'])
else:
    import json
    print(json.dumps(data, indent=2)[:1000])
