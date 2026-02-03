from retrieval.bm25_retriever import BM25Retriever, load_chunks
from retrieval.faiss_retriever import FAISSRetriever

def reciprocal_rank_fusion(results_list, k=60):
    """
    results_list: list of retriever results
                  each item is a list of dicts with 'chunk_id' and 'rank'
    """
    fused_scores = {}

    for results in results_list:
        for item in results:
            doc_id = item["chunk_id"]
            rank = item["rank"]

            if doc_id not in fused_scores:
                fused_scores[doc_id] = {
                    "score": 0,
                    "doc": item
                }

            fused_scores[doc_id]["score"] += 1 / (k + rank)

    # Sort by RRF score
    ranked_docs = sorted(
        fused_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked_docs

class HybridRetriever:
    def __init__(self):
        docs = load_chunks()
        self.bm25 = BM25Retriever(docs)
        self.faiss = FAISSRetriever()

    def search(self, query, k=5):
        bm25_results = self.bm25.search(query, k)
        faiss_results = self.faiss.search(query, k)

        fused = reciprocal_rank_fusion(
            [bm25_results, faiss_results]
        )

        final_results = []
        for rank, item in enumerate(fused[:k], start=1):
            doc = item["doc"]
            final_results.append({
                "rank": rank,
                "rrf_score": round(item["score"], 6),
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "language": doc["language"],
                "source_file": doc["source_file"],
                "source_path": doc["source_path"]
            })

        return final_results

if __name__ == "__main__":
    retriever = HybridRetriever()

    query = "Difference between ArrayList and LinkedList"
    results = retriever.search(query, k=5)

    for r in results:
        print("\nRank:", r["rank"])
        print("RRF Score:", r["rrf_score"])
        print("Source:", r["source_file"])
        print(r["text"][:300])
