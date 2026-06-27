import requests, json

# Count total
total = requests.get('http://localhost:9200/anomaly-stream/_count').json()
print(f'Total docs in anomaly-stream: {total["count"]}')

# Anomaly count
q = requests.post('http://localhost:9200/anomaly-stream/_search', json={
    'size': 0,
    'query': {'term': {'is_anomaly': True}},
}).json()
anom = q['hits']['total']['value']
print(f'Anomalies: {anom} ({100*anom/max(total["count"],1):.1f}%)')

# Attack types
q2 = requests.post('http://localhost:9200/anomaly-stream/_search', json={
    'size': 0,
    'aggs': {
        'types': {
            'terms': {'field': 'attack_type', 'size': 10}
        }
    }
}).json()
print('\nAttack type distribution:')
for b in q2['aggregations']['types']['buckets']:
    print(f'  {b["key"]}: {b["doc_count"]}')

# Sample results
q3 = requests.post('http://localhost:9200/anomaly-stream/_search', json={
    'size': 5,
    'query': {'term': {'is_anomaly': True}},
    'sort': [{'combined_score': {'order': 'desc'}}]
}).json()
print('\nTop 5 anomalies (highest combined score):')
for h in q3['hits']['hits']:
    s = h['_source']
    print(f'  icao24={s["icao24"]} attack={s["attack_type"]} recon={s["recon_error"]:.4f} score={s["combined_score"]:.4f}')
