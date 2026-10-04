import networkx as nx
import pytest

from lineage_robustness.metrics import lineage_error_amplification, set_distortion
from lineage_robustness.projections import add_false_lineage_edges, remove_lineage_edges


def sample_graph():
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    return g


def test_zero_count_is_identity():
    g = sample_graph()
    p, delta = remove_lineage_edges(g, count=0, seed=7)
    assert set(p.edges()) == set(g.edges())
    assert delta.removed == ()


def test_missing_projection_is_reproducible():
    g = sample_graph()
    p1, d1 = remove_lineage_edges(g, count=1, seed=11)
    p2, d2 = remove_lineage_edges(g, count=1, seed=11)
    assert set(p1.edges()) == set(p2.edges())
    assert d1 == d2
    assert len(d1.removed) == 1


def test_false_projection_never_adds_existing_edge():
    g = sample_graph()
    p, delta = add_false_lineage_edges(
        g,
        count=1,
        seed=5,
        candidate_edges=[("A", "B"), ("A", "C"), ("D", "A")],
    )
    assert ("A", "B") not in delta.added
    assert set(g.edges()).issubset(set(p.edges()))
    assert nx.is_directed_acyclic_graph(p)


def test_set_distortion_counts_both_directions():
    result = set_distortion(["B", "C"], ["B", "D"])
    assert result["distorted"] is True
    assert result["false_positive_elements"] == 1
    assert result["false_negative_elements"] == 1
    assert result["incorrect_elements"] == 2


def test_lea():
    assert lineage_error_amplification(
        incorrect_answer_elements=6,
        incorrect_lineage_edges=2,
    ) == 3.0
    assert lineage_error_amplification(
        incorrect_answer_elements=0,
        incorrect_lineage_edges=0,
    ) == 0.0
    assert lineage_error_amplification(
        incorrect_answer_elements=1,
        incorrect_lineage_edges=0,
    ) == float("inf")


def test_invalid_count_rejected():
    with pytest.raises(ValueError):
        remove_lineage_edges(sample_graph(), count=-1, seed=1)


def test_oversized_count_rejected():
    with pytest.raises(ValueError):
        remove_lineage_edges(sample_graph(), count=4, seed=1)

