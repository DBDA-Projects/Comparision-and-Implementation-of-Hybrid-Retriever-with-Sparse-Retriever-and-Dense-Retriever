import json
import csv
from retrieval.bm25_retriever import BM25Retriever, load_chunks
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.hybrid_retriever import HybridRetriever
from evaluation.metrics import precision_at_k, recall_at_k, mrr

docs = load_chunks()

bm25 = BM25Retriever(docs)
faiss = FAISSRetriever()
hybrid = HybridRetriever()

with open("evaluation/queries.json") as f:
    queries = json.load(f)

with open("evaluation/results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "query", "retriever", "precision@5", "recall@5", "mrr"
    ])

    for q in queries:
        query = q["query"]
        relevant = q["relevant_sources"]

        for name, retriever in [
            ("BM25", bm25),
            ("FAISS", faiss),
            ("HYBRID", hybrid)
        ]:
            results = retriever.search(query, k=5)

            writer.writerow([
                query,
                name,
                precision_at_k(results, relevant, 5),
                recall_at_k(results, relevant, 5),
                mrr(results, relevant)
            ])
