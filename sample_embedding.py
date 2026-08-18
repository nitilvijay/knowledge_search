from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

sentences = [
    "Outbreak of yellow fever reported in 1898.",
    "Epidemic disease spreads through the city in the late 19th century."
]

embeddings = model.encode(sentences)
similarity = util.cos_sim(embeddings[0], embeddings[1])
print(f"Similarity Score: {similarity.item():.4f}")