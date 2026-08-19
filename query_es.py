from elasticsearch import Elasticsearch
from PIL import Image
from sentence_transformers import SentenceTransformer

ARTICLE_INDEX = "articles"
CHUNK_INDEX = "article_chunks"
LEXICAL_LIMIT = 10
SEMANTIC_LIMIT = 30
RRF_K = 60
EMBED_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")

es = Elasticsearch("http://localhost:9200")


def open_img(img_path):
    img = Image.open(img_path)
    img.show()

def lexical_search(query):
    response = es.search(
        index=ARTICLE_INDEX,
        size=LEXICAL_LIMIT,
        query={
            "match": {
                "body": query
            }
        },
    )

    results = []
    for rank, hit in enumerate(response["hits"]["hits"], start=1):
        source = hit.get("_source", {})
        img_path = source.get("img_path")
        if not img_path:
            continue

        results.append({
            "img_path": img_path,
            "score": hit.get("_score", 0),
            "rank": rank,
        })

    return results


def semantic_search(query):
    query_embedding = EMBED_MODEL.encode(query).tolist()
    response = es.search(
        index=CHUNK_INDEX,
        size=SEMANTIC_LIMIT,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": SEMANTIC_LIMIT,
            "num_candidates": SEMANTIC_LIMIT,
        },
        source_includes=["img_path", "chunk_id"],
    )

    best_by_image = {}
    for hit in response["hits"]["hits"]:
        source = hit.get("_source", {})
        img_path = source.get("img_path")
        if not img_path:
            continue

        score = hit.get("_score", 0)
        current = best_by_image.get(img_path)
        if current is None or score > current["score"]:
            best_by_image[img_path] = {
                "img_path": img_path,
                "chunk_id": source.get("chunk_id"),
                "score": score,
            }

    results = sorted(best_by_image.values(), key=lambda item: item["score"], reverse=True)
    return results


def rrf_fusion(lexical_results, semantic_results):
    lexical_rank = {item["img_path"]: index + 1 for index, item in enumerate(lexical_results)}
    semantic_rank = {item["img_path"]: index + 1 for index, item in enumerate(semantic_results)}

    fused = {}
    for img_path in set(lexical_rank) | set(semantic_rank):
        score = 0
        if img_path in lexical_rank:
            score += 1 / (RRF_K + lexical_rank[img_path])
        if img_path in semantic_rank:
            score += 1 / (RRF_K + semantic_rank[img_path])

        fused[img_path] = {
            "img_path": img_path,
            "lexical_rank": lexical_rank.get(img_path),
            "semantic_rank": semantic_rank.get(img_path),
            "rrf_score": score,
        }

    return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)


query = input("What do you want to search: ").strip()
if not query:
    raise ValueError("Query cannot be empty")

lexical_results = lexical_search(query)
semantic_results = semantic_search(query)

if len(semantic_results) > len(lexical_results):
    semantic_results = semantic_results[:len(lexical_results)]

print(f"Lexical Results: {len(lexical_results)} | Semantic Results: {len(semantic_results)}")

final_results = rrf_fusion(lexical_results, semantic_results)
print(f"Final Results after RRF Fusion: {len(final_results)}")

print("Search results:")
for hit in final_results:
    img_path = hit["img_path"]
    print(
        f"Image Path: {img_path} | lexical_rank={hit['lexical_rank']} | semantic_rank={hit['semantic_rank']} | rrf_score={hit['rrf_score']:.6f}"
    )
    open_img(img_path)
    input("Press Enter to continue to the next image...")