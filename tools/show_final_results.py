import requests

# Count stats
total = requests.get('http://localhost:9200/anomaly-stream/_count').json()['count']

a = requests.post('http://localhost:9200/anomaly-stream/_search',
    json={'size': 0, 'query': {'term': {'is_anomaly': True}}}).json()
anom = a['hits']['total']['value']

print(f'Total windows: {total}')
print(f'Anomalies: {anom} ({100*anom/max(total,1):.1f}%)')
print(f'Normal: {total - anom} ({100*(total-anom)/max(total,1):.1f}%)')
print()

# Attack type distribution
at = requests.post('http://localhost:9200/anomaly-stream/_search',
    json={'size': 0, 'aggs': {
        'types': {'terms': {'field': 'attack_type', 'size': 10}}
    }}).json()
print('Anomaly types:')
for b in at['aggregations']['types']['buckets']:
    print(f'  {b["key"]}: {b["doc_count"]}')

# Sample
print()
s = requests.post('http://localhost:9200/anomaly-stream/_search',
    json={'size': 5, 'query': {'term': {'is_anomaly': True}}, 'sort': [{'combined_score': {'order': 'desc'}}]}
).json()
print('Top anomalies:')
for h in s['hits']['hits']:
    src = h['_source']
    print(f'  {src["icao24"]} | {src["attack_type"]:25s} | recon={src["recon_error"]:.2f} | combined={src["combined_score"]:.2f}')
