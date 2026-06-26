# test_es.py

from elasticsearch import Elasticsearch

es = Elasticsearch("http://100.99.130.69:9200")

print(es.info())