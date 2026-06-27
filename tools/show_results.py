import requests

total = requests.get('http://localhost:9200/anomaly-stream/_count')
print('Total docs:', total.json()['count'])

anom = requests.post('http://localhost:9200/anomaly-stream/_search',
    json={'size': 0, 'query': {'term': {'is_anomaly': True}}})
a = anom.json()['hits']['total']['value']
print(f'Anomalies: {a} ({100*a/total.json()["count"]:.1f}%)')

sample = requests.post('http://localhost:9200/anomaly-stream/_search',
    json={'size': 5, 'query': {'term': {'is_anomaly': True}}, 'sort': [{'combined_score': {'order': 'desc'}}]})

print('\nTop anomalies:')
for h in sample.json()['hits']['hits']:
    s = h['_source']
    print(f'  {s["icao24"]} | {s["attack_type"]:25s} | recon={s["recon_error"]:.4f} | svdd={s["svdd_distance"]:.4f} | combined={s["combined_score"]:.4f}')
