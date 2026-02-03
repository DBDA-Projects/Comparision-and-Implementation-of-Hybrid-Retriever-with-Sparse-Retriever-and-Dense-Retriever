import json
import nltk
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

nltk.download("punkt")

def load_chunks(path="data/processed_docs/python_java_chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents

        # Tokenize documents
        self.tokenized_docs = [
            word_tokenize(doc["text"].lower())
            for doc in documents
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query, k=5):
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)

        # Rank documents
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            results.append({
                "rank": rank,
                "score": scores[idx],
                "chunk_id": self.documents[idx]["chunk_id"],
                "text": self.documents[idx]["text"],
                "language": self.documents[idx]["language"],
                "source_file": self.documents[idx]["source_file"],
                "source_path": self.documents[idx]["source_path"]
            })

        return results

if __name__ == "__main__":
    docs = load_chunks()
    retriever = BM25Retriever(docs)

    query = "What is list comprehension in Python?"
    results = retriever.search(query, k=3)

    for res in results:
        print("\nRank:", res["rank"])
        print("Score:", round(res["score"], 2))
        print("Source:", res["source_file"])
        print(res["text"][:300])
