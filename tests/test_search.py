from energy_agent.rag.search import reciprocal_rank_fusion


def test_rrf_prefers_items_found_by_both_retrievers():
    vector = ["a", "b", "c"]
    text = ["d", "c", "e"]
    scores = reciprocal_rank_fusion([vector, text])
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    assert ranked[0] == "c"  # rank 3 + rank 2 beats a single rank 1
    assert set(ranked) == {"a", "b", "c", "d", "e"}


def test_rrf_ties_for_same_rank_in_different_lists():
    scores = reciprocal_rank_fusion([["a"], ["b"]])
    assert scores["a"] == scores["b"] == 1 / 61
