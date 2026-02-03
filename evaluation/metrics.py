# evaluation/metrics.py

def precision_at_k(results, relevant_sources, k):
    """
    results: list of retrieved docs
    relevant_sources: list of relevant source_file names
    """
    retrieved = results[:k]
    relevant_count = sum(
        1 for r in retrieved
        if r["source_file"] in relevant_sources
    )
    return relevant_count / k if k > 0 else 0.0


def recall_at_k(results, relevant_sources, k):
    retrieved_sources = {
        r["source_file"] for r in results[:k]
    }
    relevant_set = set(relevant_sources)

    if not relevant_set:
        return 0.0

    return len(retrieved_sources & relevant_set) / len(relevant_set)


def mrr(results, relevant_sources):
    """
    Mean Reciprocal Rank
    """
    for r in results:
        if r["source_file"] in relevant_sources:
            return 1.0 / r["rank"]
    return 0.0
