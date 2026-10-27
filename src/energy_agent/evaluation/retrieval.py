"""Retrieval-only evaluation: is the expected section among the top-k passages?

Runs without an LLM, so it isolates the search quality from the agent: hit@k is the share of
questions with a relevant passage in the top k, MRR the mean of 1/rank of the first one.
"""

from energy_agent.evaluation.golden import GoldenCase
from energy_agent.rag.search import Retriever, SearchMode


def first_relevant_rank(
    retriever: Retriever, case: GoldenCase, mode: SearchMode, k: int
) -> int | None:
    results = retriever.search(case.question, k=k, mode=mode)
    for rank, result in enumerate(results, start=1):
        if any(section in result.section for section in case.section):
            return rank
    return None


def retrieval_metrics(
    retriever: Retriever,
    cases: list[GoldenCase],
    modes: tuple[SearchMode, ...] = ("vector", "text", "hybrid"),
    k: int = 5,
) -> dict[str, dict[str, float]]:
    cases = [c for c in cases if c.section]
    metrics = {}
    for mode in modes:
        ranks = [first_relevant_rank(retriever, case, mode, k) for case in cases]
        metrics[mode] = {
            "hit@1": round(sum(r == 1 for r in ranks) / len(ranks), 3),
            "hit@3": round(sum(r is not None and r <= 3 for r in ranks) / len(ranks), 3),
            f"hit@{k}": round(sum(r is not None for r in ranks) / len(ranks), 3),
            "mrr": round(sum(1 / r for r in ranks if r) / len(ranks), 3),
        }
    return metrics
