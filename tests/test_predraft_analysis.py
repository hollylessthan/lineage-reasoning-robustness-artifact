import networkx as nx

from lineage_robustness.spider_replication import (
    SpiderModelGraph,
    deletion_nonlocal_severity,
    model_depth,
    to_nx,
)
from lineage_robustness.predraft_analysis import (
    closure_nonlocal_severity,
    lemma_diagnostics,
    longest_path_levels,
    matched_layered_null,
    path_multiplicity_fraction,
)


def chain_graph():
    return SpiderModelGraph(
        project="chain",
        names=("a", "b", "c", "d"),
        edges=frozenset({(0, 1), (1, 2), (2, 3)}),
        unresolved_refs=0,
        duplicate_model_names=0,
    )


def diamond_graph():
    return SpiderModelGraph(
        project="diamond",
        names=("a", "b", "c", "d"),
        edges=frozenset({(0, 1), (0, 2), (1, 3), (2, 3)}),
        unresolved_refs=0,
        duplicate_model_names=0,
    )


def test_matched_null_preserves_size_edges_and_depth():
    g = chain_graph()
    n = matched_layered_null(g, seed=7)
    assert n.n == g.n
    assert len(n.edges) == len(g.edges)
    assert model_depth(n) == model_depth(g)


def test_chain_has_no_multi_path_pairs():
    r = path_multiplicity_fraction(chain_graph())
    assert r["multi_path_fraction"] == 0.0


def test_diamond_has_multi_path_pair():
    r = path_multiplicity_fraction(diamond_graph())
    assert r["multi_path_pairs"] >= 1
    assert r["multi_path_fraction"] > 0.0


def test_unique_path_chain_satisfies_lemma_and_endpoint_identity():
    r = lemma_diagnostics(chain_graph())
    assert r["all_bounds_hold"] is True
    assert r["all_endpoint_identities_hold"] is True


def test_redundant_diamond_satisfies_general_bound():
    r = lemma_diagnostics(diamond_graph())
    assert r["all_bounds_hold"] is True
    assert r["all_endpoint_identities_hold"] is True


def uneven_layer_graph():
    # layer 0: a, b; layer 1: c; layer 2: d, e; layer 3: f
    return SpiderModelGraph(
        project="uneven",
        names=("a", "b", "c", "d", "e", "f"),
        edges=frozenset({(0, 2), (1, 2), (2, 3), (2, 4), (0, 4), (3, 5), (1, 5)}),
        unresolved_refs=0,
        duplicate_model_names=0,
    )


def test_matched_null_preserves_every_longest_path_layer():
    g = uneven_layer_graph()
    observed = longest_path_levels(to_nx(g))
    for seed in range(50):
        n = matched_layered_null(g, seed=seed)
        assert len(n.edges) == len(g.edges)
        assert longest_path_levels(to_nx(n)) == observed


def test_closure_severity_matches_probe_severity():
    for g in (chain_graph(), diamond_graph(), uneven_layer_graph()):
        for edge in sorted(g.edges):
            expected = deletion_nonlocal_severity(g, edge)["nonlocal_incorrect_elements"]
            assert closure_nonlocal_severity(g, edge) == expected


def test_lemma_diagnostics_reports_independent_severity_check():
    r = lemma_diagnostics(uneven_layer_graph())
    assert r["all_independent_severity_matches"] is True

