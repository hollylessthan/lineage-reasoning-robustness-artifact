import networkx as nx

from lineage_robustness.dwbench import DWGraph
from lineage_robustness.phase0a import answer_case, freeze_cases, run_k1
from lineage_robustness.projections import (
    add_false_lineage_edges,
    admissible_false_edges,
    remove_lineage_edges,
)


def toy():
    return DWGraph(
        dataset="toy",
        names=("src", "mid", "fact", "child", "other"),
        fk_edges=frozenset({(3, 2)}),
        lineage_edges=frozenset({(0, 1), (1, 2)}),
    )


def test_frozen_population_is_not_regenerated_after_deletion():
    graph = toy()
    cases = freeze_cases(graph)
    trans = next(c for c in cases if c.subtype == "transitive")
    projected = DWGraph(
        dataset=graph.dataset,
        names=graph.names,
        fk_edges=graph.fk_edges,
        lineage_edges=frozenset({(0, 1)}),
    )
    assert set(answer_case(projected, trans)) != set(trans.gold)
    assert len(cases) == len(freeze_cases(graph))


def test_admissible_false_edges_preserve_dag_and_layer_direction():
    g = nx.DiGraph([(0, 1), (1, 2)])
    g.add_node(3)
    candidates = admissible_false_edges(g)
    for edge in candidates:
        p = g.copy()
        p.add_edge(*edge)
        assert nx.is_directed_acyclic_graph(p)
    assert (2, 0) not in candidates
    assert (0, 2) in candidates


def test_count_based_projection_exact_and_nonmutating():
    g = nx.DiGraph([(0, 1), (1, 2)])
    original = set(g.edges())

    p, delta = remove_lineage_edges(g, count=1, seed=7)
    assert len(delta.removed) == 1
    assert set(g.edges()) == original
    assert p.number_of_edges() == 1

    p2, delta2 = add_false_lineage_edges(g, count=1, seed=7)
    assert len(delta2.added) == 1
    assert set(g.edges()) == original
    assert nx.is_directed_acyclic_graph(p2)


def test_k1_enumerates_every_true_deletion():
    result = run_k1(toy())
    assert result["deletions"]["edits"] == 2
    assert result["additions"]["edits"] >= 1

