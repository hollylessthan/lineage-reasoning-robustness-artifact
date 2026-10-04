from lineage_robustness.dwbench import DWGraph
from lineage_robustness.phase0b import (
    answer_operation,
    evaluate_edit,
    freeze_anchor_complete_probes,
)


def toy():
    return DWGraph(
        dataset="toy",
        names=("src", "mid", "fact", "isolated"),
        fk_edges=frozenset(),
        lineage_edges=frozenset({(0, 1), (1, 2)}),
    )


def test_anchor_complete_includes_empty_answers():
    g = toy()
    probes = freeze_anchor_complete_probes(g)
    assert len(probes) == 5 * g.n
    empty = [
        p for p in probes
        if p.anchor == 3 and p.operation == "transitive_downstream"
    ]
    assert len(empty) == 1
    assert empty[0].gold == ()


def test_deletion_has_local_and_propagated_damage():
    g = toy()
    probes = freeze_anchor_complete_probes(g)
    projected = DWGraph(
        dataset=g.dataset,
        names=g.names,
        fk_edges=g.fk_edges,
        lineage_edges=frozenset({(1, 2)}),
    )
    result = evaluate_edit(g, projected, probes, (0, 1))
    assert result["local"]["distorted"] == 2
    assert result["propagated"]["distorted"] > 0
    assert result["potential_nonlocal_exposure"] > 0


def test_false_edge_can_distort_null_gold_probe():
    g = toy()
    probes = freeze_anchor_complete_probes(g)
    projected = DWGraph(
        dataset=g.dataset,
        names=g.names,
        fk_edges=g.fk_edges,
        lineage_edges=frozenset({(0, 1), (1, 2), (0, 2)}),
    )
    # Shortcut does not create null-gold damage in this toy, so add an edge
    # from the otherwise isolated node into the lineage chain.
    projected2 = DWGraph(
        dataset=g.dataset,
        names=g.names,
        fk_edges=g.fk_edges,
        lineage_edges=frozenset({(0, 1), (1, 2), (3, 2)}),
    )
    result = evaluate_edit(g, projected2, probes, (3, 2))
    assert result["null_gold"]["distorted"] > 0

