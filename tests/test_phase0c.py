from lineage_robustness.dwbench import DWGraph
from lineage_robustness.phase0b import evaluate_edit, freeze_anchor_complete_probes
from lineage_robustness.phase0c import exact_nonlocal_oracle, gini, top_fraction_share


def toy():
    return DWGraph(
        dataset="toy",
        names=("a", "b", "c", "d"),
        fk_edges=frozenset({(3, 2)}),
        lineage_edges=frozenset({(0, 1), (1, 2)}),
    )


def test_exact_oracle_matches_harness_for_deletion():
    g = toy()
    projected = DWGraph(
        dataset=g.dataset,
        names=g.names,
        fk_edges=g.fk_edges,
        lineage_edges=frozenset({(1, 2)}),
    )
    probes = freeze_anchor_complete_probes(g)
    actual = evaluate_edit(g, projected, probes, (0, 1))["propagated"]
    oracle = exact_nonlocal_oracle(g, projected, (0, 1))
    for key in ("distorted", "fp", "fn", "incorrect_elements"):
        assert int(actual[key]) == int(oracle[key])


def test_exact_oracle_matches_harness_for_addition():
    g = toy()
    projected = DWGraph(
        dataset=g.dataset,
        names=g.names,
        fk_edges=g.fk_edges,
        lineage_edges=frozenset({(0, 1), (1, 2), (0, 2)}),
    )
    probes = freeze_anchor_complete_probes(g)
    actual = evaluate_edit(g, projected, probes, (0, 2))["propagated"]
    oracle = exact_nonlocal_oracle(g, projected, (0, 2))
    for key in ("distorted", "fp", "fn", "incorrect_elements"):
        assert int(actual[key]) == int(oracle[key])


def test_concentration_helpers():
    assert gini([0, 0, 0]) == 0.0
    assert 0.0 <= gini([1, 1, 8]) <= 1.0
    assert top_fraction_share([1, 1, 8], 0.10) == 0.8

