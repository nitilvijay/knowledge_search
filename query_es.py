from elasticsearch import Elasticsearch
from networkx import hits
from PIL import Image

es = Elasticsearch("http://localhost:9200")


def open_img(img_path):
    img = Image.open(img_path)
    img.show()

response = es.search(
    index="articles",
    query={
        "match": {
            "body": "nitrogen"
        }
    }
)

# print(response)
# print(response['hits'])

print("Search results:")
for hit in response['hits']['hits']:
    img_path = hit['_source']["img_path"]
    print(f"Image Path: {img_path}")
    open_img(img_path)
    input("Press Enter to continue to the next image...")