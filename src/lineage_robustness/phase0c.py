from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx

from lineage_robustness.dwbench import DWGraph, load_pyg_graph, trace_forward
from lineage_robustness.phase0b import evaluate_edit, freeze_anchor_complete_probes
from lineage_robustness.projections import admissible_false_edges


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


def _with_lineage(graph: DWGraph, edges) -> DWGraph:
    return DWGraph(graph.dataset, graph.names, graph.fk_edges, frozenset(edges))


def descendants(graph: DWGraph, anchor: int) -> set[int]:
    return trace_forward(_adj(graph.n, graph.lineage_edges), anchor)


def ancestors(graph: DWGraph, anchor: int) -> set[int]:
    return trace_forward(_rev(graph.n, graph.lineage_edges), anchor)


def combined_answer_ids(graph: DWGraph, anchor: int) -> set[int]:
    reached = descendants(graph, anchor)
    scope = reached | {anchor}
    fkrev = _rev(graph.n, graph.fk_edges)
    fk_affected = {
        child
        for node in scope
        for child in fkrev[node]
        if child not in scope
    }
    return reached | fk_affected


def exact_nonlocal_oracle(
    pristine: DWGraph,
    projected: DWGraph,
    edge: tuple[int, int],
) -> dict[str, int]:
    """Exact nonlocal probe damage from graph semantics.

    This oracle does not iterate through the frozen Probe objects. It derives
    expected set differences directly from reachability and combined-impact
    semantics for every anchor, excluding edited-edge endpoints.
    """
    u, v = edge
    fp = fn = distorted = incorrect = 0

    # Transitive downstream: exclude source endpoint u.
    for anchor in range(pristine.n):
        if anchor == u:
            continue
        g = descendants(pristine, anchor)
        o = descendants(projected, anchor)
        d_fp = len(o - g)
        d_fn = len(g - o)
        fp += d_fp
        fn += d_fn
        incorrect += d_fp + d_fn
        distorted += int(g != o)

    # Transitive upstream: exclude target endpoint v.
    for anchor in range(pristine.n):
        if anchor == v:
            continue
        g = ancestors(pristine, anchor)
        o = ancestors(projected, anchor)
        d_fp = len(o - g)
        d_fn = len(g - o)
        fp += d_fp
        fn += d_fn
        incorrect += d_fp + d_fn
        distorted += int(g != o)

    # Combined impact: exclude source endpoint u.
    for anchor in range(pristine.n):
        if anchor == u:
            continue
        g = combined_answer_ids(pristine, anchor)
        o = combined_answer_ids(projected, anchor)
        d_fp = len(o - g)
        d_fn = len(g - o)
        fp += d_fp
        fn += d_fn
        incorrect += d_fp + d_fn
        distorted += int(g != o)

    return {
        "distorted": distorted,
        "fp": fp,
        "fn": fn,
        "incorrect_elements": incorrect,
    }


def graph_diagnostics(graph: DWGraph) -> dict[str, float | int]:
    lg = nx.DiGraph()
    lg.add_nodes_from(range(graph.n))
    lg.add_edges_from(graph.lineage_edges)
    if not nx.is_directed_acyclic_graph(lg):
        raise ValueError(f"{graph.dataset} lineage graph is not a DAG")

    desc_sizes = [len(nx.descendants(lg, node)) for node in lg.nodes]
    candidates = admissible_false_edges(lg)
    shortcut_count = sum(1 for u, v in candidates if nx.has_path(lg, u, v))
    possible_fk = graph.n * (graph.n - 1)

    return {
        "tables": graph.n,
        "lineage_edges": len(graph.lineage_edges),
        "fk_edges": len(graph.fk_edges),
        "lineage_longest_path": nx.algorithms.dag.dag_longest_path_length(lg),
        "mean_descendants": sum(desc_sizes) / len(desc_sizes) if desc_sizes else 0.0,
        "fk_density": len(graph.fk_edges) / possible_fk if possible_fk else 0.0,
        "admissible_false_additions": len(candidates),
        "shortcut_additions": shortcut_count,
        "shortcut_fraction": shortcut_count / len(candidates) if candidates else 0.0,
    }


def gini(values: list[float]) -> float:
    vals = sorted(max(0.0, float(v)) for v in values)
    if not vals or sum(vals) == 0:
        return 0.0
    n = len(vals)
    total = sum(vals)
    weighted = sum((i + 1) * v for i, v in enumerate(vals))
    return (2 * weighted) / (n * total) - (n + 1) / n


def top_fraction_share(values: list[float], fraction: float = 0.10) -> float:
    vals = sorted((max(0.0, float(v)) for v in values), reverse=True)
    total = sum(vals)
    if not vals or total == 0:
        return 0.0
    k = max(1, math.ceil(fraction * len(vals)))
    return sum(vals[:k]) / total


def run_dataset(graph: DWGraph) -> dict[str, object]:
    probes = freeze_anchor_complete_probes(graph)
    lg = nx.DiGraph()
    lg.add_nodes_from(range(graph.n))
    lg.add_edges_from(graph.lineage_edges)

    rows = []
    oracle_mismatches = []

    def add_row(kind: str, edge: tuple[int, int], projected: DWGraph) -> None:
        actual = evaluate_edit(graph, projected, probes, edge)["propagated"]
        oracle = exact_nonlocal_oracle(graph, projected, edge)
        matches = all(int(actual[k]) == int(oracle[k]) for k in oracle)

        alternate_path = False
        preexisting_reachability = False
        if kind == "delete":
            trial = lg.copy()
            trial.remove_edge(*edge)
            alternate_path = nx.has_path(trial, *edge)
        else:
            preexisting_reachability = nx.has_path(lg, *edge)

        row = {
            "kind": kind,
            "edge_index": list(edge),
            "edge": [graph.names[edge[0]], graph.names[edge[1]]],
            "actual_nonlocal": {
                k: int(actual[k]) for k in ("distorted", "fp", "fn", "incorrect_elements")
            },
            "oracle_nonlocal": oracle,
            "oracle_match": matches,
            "alternate_path": alternate_path,
            "preexisting_reachability": preexisting_reachability,
        }
        rows.append(row)
        if not matches:
            oracle_mismatches.append(row)

    for edge in sorted(graph.lineage_edges):
        edges = set(graph.lineage_edges)
        edges.remove(edge)
        add_row("delete", edge, _with_lineage(graph, edges))

    for edge in admissible_false_edges(lg):
        edges = set(graph.lineage_edges)
        edges.add(edge)
        add_row("add", edge, _with_lineage(graph, edges))

    deletions = [r for r in rows if r["kind"] == "delete"]
    additions = [r for r in rows if r["kind"] == "add"]

    redundant = [r for r in deletions if r["alternate_path"]]
    shortcuts = [r for r in additions if r["preexisting_reachability"]]

    def severity_summary(selected: list[dict]) -> dict[str, float | int]:
        vals = [r["oracle_nonlocal"]["incorrect_elements"] for r in selected]
        return {
            "edits": len(selected),
            "zero_nonlocal_damage": sum(v == 0 for v in vals),
            "mean_nonlocal_incorrect_elements": sum(vals) / len(vals) if vals else 0.0,
            "max_nonlocal_incorrect_elements": max(vals, default=0),
            "gini_nonlocal_severity": gini(vals),
            "top_10pct_share_nonlocal_severity": top_fraction_share(vals, 0.10),
        }

    return {
        "dataset": graph.dataset,
        "graph_diagnostics": graph_diagnostics(graph),
        "oracle": {
            "edits_checked": len(rows),
            "matches": len(rows) - len(oracle_mismatches),
            "mismatches": len(oracle_mismatches),
            "exact": len(oracle_mismatches) == 0,
            "mismatch_examples": oracle_mismatches[:10],
        },
        "deletions": {
            **severity_summary(deletions),
            "redundant_edges": len(redundant),
            "redundant_edges_with_nonzero_nonlocal_damage": sum(
                r["oracle_nonlocal"]["incorrect_elements"] > 0 for r in redundant
            ),
        },
        "additions": {
            **severity_summary(additions),
            "shortcut_additions": len(shortcuts),
            "shortcut_additions_with_nonzero_nonlocal_damage": sum(
                r["oracle_nonlocal"]["incorrect_elements"] > 0 for r in shortcuts
            ),
        },
        "per_edit": rows,
    }


def run_paths(paths: list[Path], output: Path) -> dict[str, object]:
    payload = {
        "phase": "0C-exact-severity-oracle",
        "datasets": [run_dataset(load_pyg_graph(path)) for path in paths],
    }
    payload["all_oracles_exact"] = all(d["oracle"]["exact"] for d in payload["datasets"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload

