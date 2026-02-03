import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_PATH = "data/processed_docs/python_java_chunks.json"
FAISS_DIR = "data/faiss"
INDEX_PATH = f"{FAISS_DIR}/index.faiss"
META_PATH = f"{FAISS_DIR}/metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_chunks():
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_faiss_index():
    docs = load_chunks()
    texts = [doc["text"] for doc in docs]

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))

    os.makedirs(FAISS_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    # Save metadata separately
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)

    print(f"FAISS index built with {len(docs)} vectors")

class FAISSRetriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = faiss.read_index(INDEX_PATH)

        with open(META_PATH, "r", encoding="utf-8") as f:
            self.docs = json.load(f)

    def search(self, query, k=5):
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        scores, indices = self.index.search(query_emb, k)

        results = []
        for rank, idx in enumerate(indices[0], start=1):
            results.append({
                "rank": rank,
                "score": float(scores[0][rank - 1]),
                "chunk_id": self.docs[idx]["chunk_id"],
                "text": self.docs[idx]["text"],
                "language": self.docs[idx]["language"],
                "source_file": self.docs[idx]["source_file"],
                "source_path": self.docs[idx]["source_path"]
            })

        return results

if __name__ == "__main__":
    build_faiss_index()
