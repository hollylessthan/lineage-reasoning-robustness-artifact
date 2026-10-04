from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import networkx as nx

from lineage_robustness.dwbench import DWGraph, load_pyg_graph, trace_forward
from lineage_robustness.projections import admissible_false_edges


@dataclass(frozen=True)
class Probe:
    probe_id: str
    operation: str
    anchor: int
    gold: tuple[str, ...]


def _adj(n: int, edges) -> dict[int, set[int]]:
    out = {i: set() for i in range(n)}
    for u, v in edges:
        out[u].add(v)
    return out


def _rev(n: int, edges) -> dict[int, set[int]]:
    out = {i: set() for i in range(n)}
    for u, v in edges:
        out[v].add(u)
    return out


def answer_operation(graph: DWGraph, operation: str, anchor: int) -> tuple[str, ...]:
    names = graph.names
    ladj = _adj(graph.n, graph.lineage_edges)
    lrev = _rev(graph.n, graph.lineage_edges)
    fkrev = _rev(graph.n, graph.fk_edges)

    if operation == "direct_downstream":
        nodes = ladj[anchor]
    elif operation == "direct_upstream":
        nodes = lrev[anchor]
    elif operation == "transitive_downstream":
        nodes = trace_forward(ladj, anchor)
    elif operation == "transitive_upstream":
        nodes = trace_forward(lrev, anchor)
    elif operation == "combined_impact":
        reached = trace_forward(ladj, anchor)
        scope = reached | {anchor}
        fk_affected = {
            child
            for node in scope
            for child in fkrev[node]
            if child not in scope
        }
        nodes = reached | fk_affected
    else:
        raise ValueError(f"unsupported operation: {operation}")

    return tuple(sorted(names[i] for i in nodes))


def freeze_anchor_complete_probes(graph: DWGraph) -> tuple[Probe, ...]:
    operations = (
        "direct_downstream",
        "direct_upstream",
        "transitive_downstream",
        "transitive_upstream",
        "combined_impact",
    )
    probes = []
    for anchor in range(graph.n):
        for operation in operations:
            probes.append(
                Probe(
                    probe_id=f"{graph.dataset}:{operation}:{anchor}",
                    operation=operation,
                    anchor=anchor,
                    gold=answer_operation(graph, operation, anchor),
                )
            )
    return tuple(probes)


def _set_metrics(gold: tuple[str, ...], observed: tuple[str, ...]) -> dict[str, float | int | bool]:
    g, o = set(gold), set(observed)
    fp = len(o - g)
    fn = len(g - o)
    tp = len(g & o)
    precision = 1.0 if not g and not o else (tp / len(o) if o else 0.0)
    recall = 1.0 if not g else tp / len(g)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "exact": g == o,
        "fp": fp,
        "fn": fn,
        "incorrect_elements": fp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _with_lineage(graph: DWGraph, edges) -> DWGraph:
    return DWGraph(
        dataset=graph.dataset,
        names=graph.names,
        fk_edges=graph.fk_edges,
        lineage_edges=frozenset(edges),
    )


def _is_local_probe(probe: Probe, edge: tuple[int, int]) -> bool:
    u, v = edge
    return (
        (probe.operation == "direct_downstream" and probe.anchor == u)
        or (probe.operation == "direct_upstream" and probe.anchor == v)
    )


def _is_propagated_probe(probe: Probe, edge: tuple[int, int]) -> bool:
    """Nonlocal probe whose anchor is not an edited-edge endpoint."""
    u, v = edge
    if probe.operation == "transitive_downstream":
        return probe.anchor != u
    if probe.operation == "transitive_upstream":
        return probe.anchor != v
    if probe.operation == "combined_impact":
        return probe.anchor != u
    return False


def potential_nonlocal_exposure(
    graph: DWGraph,
    edge: tuple[int, int],
) -> int:
    """Count fixed nonlocal probes structurally capable of observing this edge.

    This is computed from pristine topology and endpoints only; it does not use
    observed damage.
    """
    u, v = edge
    ladj = _adj(graph.n, graph.lineage_edges)
    lrev = _rev(graph.n, graph.lineage_edges)

    upstream_anchors = trace_forward(lrev, u)
    downstream_anchors = trace_forward(ladj, v)

    # Exclude the edited endpoints themselves. These are endpoint-anchored
    # effects, not propagation beyond the edited relation.
    # Two upstream-anchor families can observe u->v: transitive-downstream
    # and combined-impact. One downstream-anchor family can observe it:
    # transitive-upstream.
    return 2 * len(upstream_anchors) + len(downstream_anchors)


def evaluate_edit(
    pristine: DWGraph,
    projected: DWGraph,
    probes: tuple[Probe, ...],
    edge: tuple[int, int],
) -> dict[str, object]:
    local = []
    propagated = []
    null_gold = []
    by_operation: dict[str, list[dict]] = {}

    for probe in probes:
        observed = answer_operation(projected, probe.operation, probe.anchor)
        metrics = _set_metrics(probe.gold, observed)
        row = {
            "probe_id": probe.probe_id,
            "operation": probe.operation,
            "anchor": probe.anchor,
            "gold_empty": len(probe.gold) == 0,
            **metrics,
        }
        by_operation.setdefault(probe.operation, []).append(row)
        if _is_local_probe(probe, edge):
            local.append(row)
        elif _is_propagated_probe(probe, edge):
            propagated.append(row)
        if row["gold_empty"]:
            null_gold.append(row)

    def summarize(rows: list[dict]) -> dict[str, float | int]:
        if not rows:
            return {
                "probes": 0,
                "distorted": 0,
                "adr": 0.0,
                "fp": 0,
                "fn": 0,
                "incorrect_elements": 0,
                "mean_f1": 1.0,
            }
        distorted = sum(not r["exact"] for r in rows)
        return {
            "probes": len(rows),
            "distorted": distorted,
            "adr": distorted / len(rows),
            "fp": sum(int(r["fp"]) for r in rows),
            "fn": sum(int(r["fn"]) for r in rows),
            "incorrect_elements": sum(int(r["incorrect_elements"]) for r in rows),
            "mean_f1": sum(float(r["f1"]) for r in rows) / len(rows),
        }

    exposure = potential_nonlocal_exposure(pristine, edge)
    propagated_summary = summarize(propagated)
    return {
        "local": summarize(local),
        "propagated": propagated_summary,
        "null_gold": summarize(null_gold),
        "potential_nonlocal_exposure": exposure,
        "propagated_damage_per_exposure": (
            propagated_summary["distorted"] / exposure if exposure else 0.0
        ),
        "by_operation": {
            op: summarize(rows)
            for op, rows in sorted(by_operation.items())
        },
    }


def run_k1_anchor_complete(graph: DWGraph) -> dict[str, object]:
    probes = freeze_anchor_complete_probes(graph)
    lg = nx.DiGraph()
    lg.add_nodes_from(range(graph.n))
    lg.add_edges_from(graph.lineage_edges)
    if not nx.is_directed_acyclic_graph(lg):
        raise ValueError(f"{graph.dataset} lineage graph is not a DAG")

    rows = []
    for edge in sorted(graph.lineage_edges):
        edges = set(graph.lineage_edges)
        edges.remove(edge)
        rows.append({
            "kind": "delete",
            "edge_index": list(edge),
            "edge": [graph.names[edge[0]], graph.names[edge[1]]],
            "metrics": evaluate_edit(
                graph, _with_lineage(graph, edges), probes, edge
            ),
        })

    for edge in admissible_false_edges(lg):
        edges = set(graph.lineage_edges)
        edges.add(edge)
        rows.append({
            "kind": "add",
            "edge_index": list(edge),
            "edge": [graph.names[edge[0]], graph.names[edge[1]]],
            "metrics": evaluate_edit(
                graph, _with_lineage(graph, edges), probes, edge
            ),
        })

    def kind_summary(kind: str) -> dict[str, object]:
        selected = [r for r in rows if r["kind"] == kind]
        prop = [r["metrics"]["propagated"] for r in selected]
        nulls = [r["metrics"]["null_gold"] for r in selected]
        expnorm = [r["metrics"]["propagated_damage_per_exposure"] for r in selected]
        return {
            "edits": len(selected),
            "nonzero_propagated_edits": sum(p["distorted"] > 0 for p in prop),
            "propagated_edit_rate": (
                sum(p["distorted"] > 0 for p in prop) / len(selected)
                if selected else 0.0
            ),
            "mean_propagated_distorted_probes": (
                sum(p["distorted"] for p in prop) / len(selected)
                if selected else 0.0
            ),
            "max_propagated_distorted_probes": max(
                (p["distorted"] for p in prop), default=0
            ),
            "mean_propagated_incorrect_elements": (
                sum(p["incorrect_elements"] for p in prop) / len(selected)
                if selected else 0.0
            ),
            "max_propagated_incorrect_elements": max(
                (p["incorrect_elements"] for p in prop), default=0
            ),
            "mean_propagated_damage_per_exposure": (
                sum(expnorm) / len(expnorm) if expnorm else 0.0
            ),
            "edits_distorting_null_gold_probes": sum(
                n["distorted"] > 0 for n in nulls
            ),
        }

    empty_by_operation = {}
    for op in {
        "direct_downstream",
        "direct_upstream",
        "transitive_downstream",
        "transitive_upstream",
        "combined_impact",
    }:
        op_probes = [p for p in probes if p.operation == op]
        empty_by_operation[op] = {
            "probes": len(op_probes),
            "empty_gold": sum(len(p.gold) == 0 for p in op_probes),
        }

    return {
        "dataset": graph.dataset,
        "tables": graph.n,
        "probes": len(probes),
        "probe_inventory": empty_by_operation,
        "deletions": kind_summary("delete"),
        "additions": kind_summary("add"),
        "per_edit": rows,
    }


def run_paths(paths: list[Path], output: Path) -> dict[str, object]:
    payload = {
        "phase": "0B-anchor-complete",
        "datasets": [
            run_k1_anchor_complete(load_pyg_graph(path))
            for path in paths
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload

