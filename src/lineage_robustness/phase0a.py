from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import networkx as nx

from lineage_robustness.dwbench import DWGraph, load_pyg_graph, trace_forward
from lineage_robustness.projections import admissible_false_edges, dag_levels


@dataclass(frozen=True)
class FrozenCase:
    case_id: str
    subtype: str
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


def freeze_cases(graph: DWGraph) -> tuple[FrozenCase, ...]:
    names = graph.names
    ladj = _adj(graph.n, graph.lineage_edges)
    lrev = _rev(graph.n, graph.lineage_edges)
    fkrev = _rev(graph.n, graph.fk_edges)
    cases: list[FrozenCase] = []
    sources = [u for u in range(graph.n) if ladj[u]]
    targets = [v for v in range(graph.n) if lrev[v]]

    for u in sources:
        cases.append(FrozenCase(f"{graph.dataset}:forward:{u}", "forward", u,
                                tuple(sorted(names[v] for v in ladj[u]))))
    for v in targets:
        cases.append(FrozenCase(f"{graph.dataset}:reverse:{v}", "reverse", v,
                                tuple(sorted(names[u] for u in lrev[v]))))
    for u in sources:
        reached = trace_forward(ladj, u)
        if reached - ladj[u] - {u}:
            cases.append(FrozenCase(f"{graph.dataset}:transitive:{u}", "transitive", u,
                                    tuple(sorted(names[v] for v in reached))))
    for u in sources:
        reached = trace_forward(ladj, u)
        scope = reached | {u}
        fk_affected = {child for node in scope for child in fkrev[node] if child not in scope}
        if fk_affected:
            cases.append(FrozenCase(f"{graph.dataset}:combined_impact:{u}", "combined_impact", u,
                                    tuple(sorted(names[v] for v in reached | fk_affected))))
    for v in targets:
        if len(lrev[v]) >= 3:
            cases.append(FrozenCase(f"{graph.dataset}:multi_source:{v}", "multi_source", v,
                                    tuple(sorted(names[u] for u in lrev[v]))))
    return tuple(cases)


def answer_case(graph: DWGraph, case: FrozenCase) -> tuple[str, ...]:
    names = graph.names
    ladj = _adj(graph.n, graph.lineage_edges)
    lrev = _rev(graph.n, graph.lineage_edges)
    fkrev = _rev(graph.n, graph.fk_edges)
    if case.subtype == "forward":
        nodes = ladj[case.anchor]
    elif case.subtype in {"reverse", "multi_source"}:
        nodes = lrev[case.anchor]
    elif case.subtype == "transitive":
        nodes = trace_forward(ladj, case.anchor)
    elif case.subtype == "combined_impact":
        reached = trace_forward(ladj, case.anchor)
        scope = reached | {case.anchor}
        nodes = reached | {child for node in scope for child in fkrev[node] if child not in scope}
    else:
        raise ValueError(case.subtype)
    return tuple(sorted(names[i] for i in nodes))


def _set_metrics(gold: tuple[str, ...], observed: tuple[str, ...]) -> dict:
    g, o = set(gold), set(observed)
    fp, fn, tp = len(o-g), len(g-o), len(g&o)
    precision = 1.0 if not g and not o else (tp / len(o) if o else 0.0)
    recall = 1.0 if not g else tp / len(g)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"exact": g == o, "fp": fp, "fn": fn, "incorrect_elements": fp + fn,
            "precision": precision, "recall": recall, "f1": f1}


def _with_lineage(graph: DWGraph, edges) -> DWGraph:
    return DWGraph(graph.dataset, graph.names, graph.fk_edges, frozenset(edges))


def structural_features(graph: DWGraph, cases: tuple[FrozenCase, ...], edge: tuple[int, int]) -> dict:
    u, v = edge
    lg = nx.DiGraph()
    lg.add_nodes_from(range(graph.n))
    lg.add_edges_from(graph.lineage_edges)
    levels = dag_levels(lg)
    ladj = _adj(graph.n, graph.lineage_edges)
    fkrev = _rev(graph.n, graph.fk_edges)
    downstream = trace_forward(ladj, v) | {v}

    upstream_anchors = 0
    reverse_anchors = 0
    for case in cases:
        if case.subtype in {"forward", "transitive", "combined_impact"}:
            if case.anchor == u or u in trace_forward(ladj, case.anchor):
                upstream_anchors += 1
        elif case.anchor == v:
            reverse_anchors += 1

    fk_fanout = len({child for node in downstream for child in fkrev[node] if child not in downstream})
    return {
        "source_level": levels[u],
        "target_level": levels[v],
        "level_gap": levels[v] - levels[u],
        "downstream_closure_size": len(downstream),
        "upstream_anchor_count": upstream_anchors,
        "reverse_anchor_count": reverse_anchors,
        "downstream_fk_fanout": fk_fanout,
    }


def evaluate_projection(projected: DWGraph, cases: tuple[FrozenCase, ...]) -> dict:
    rows = []
    for case in cases:
        m = _set_metrics(case.gold, answer_case(projected, case))
        rows.append({"subtype": case.subtype, **m})

    by_subtype = {}
    for subtype in sorted({c.subtype for c in cases}):
        subset = [r for r in rows if r["subtype"] == subtype]
        by_subtype[subtype] = {
            "cases": len(subset),
            "distorted": sum(not r["exact"] for r in subset),
            "adr": sum(not r["exact"] for r in subset) / len(subset),
            "fp": sum(r["fp"] for r in subset),
            "fn": sum(r["fn"] for r in subset),
            "mean_f1": sum(r["f1"] for r in subset) / len(subset),
        }

    distorted = sum(not r["exact"] for r in rows)
    return {
        "cases": len(rows),
        "distorted": distorted,
        "adr": distorted / len(rows),
        "fp": sum(r["fp"] for r in rows),
        "fn": sum(r["fn"] for r in rows),
        "incorrect_elements": sum(r["incorrect_elements"] for r in rows),
        "mean_f1": sum(r["f1"] for r in rows) / len(rows),
        "by_subtype": by_subtype,
    }


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"edits": 0}
    adrs = [r["damage"]["adr"] for r in rows]
    wrong = [r["damage"]["incorrect_elements"] for r in rows]
    top = sorted(rows, key=lambda r: (r["damage"]["incorrect_elements"], r["damage"]["distorted"]), reverse=True)[:10]
    return {
        "edits": len(rows),
        "zero_damage_edits": sum(x == 0 for x in wrong),
        "nonzero_damage_edits": sum(x > 0 for x in wrong),
        "mean_adr": sum(adrs) / len(adrs),
        "max_adr": max(adrs),
        "mean_incorrect_elements": sum(wrong) / len(wrong),
        "max_incorrect_elements": max(wrong),
        "top_10": top,
    }


def run_k1(graph: DWGraph) -> dict:
    cases = freeze_cases(graph)
    lg = nx.DiGraph()
    lg.add_nodes_from(range(graph.n))
    lg.add_edges_from(graph.lineage_edges)
    if not nx.is_directed_acyclic_graph(lg):
        raise ValueError(f"{graph.dataset} lineage graph is not a DAG")

    deletions = []
    for edge in sorted(graph.lineage_edges):
        edges = set(graph.lineage_edges)
        edges.remove(edge)
        deletions.append({
            "kind": "delete",
            "edge": [graph.names[edge[0]], graph.names[edge[1]]],
            "edge_index": list(edge),
            "features": structural_features(graph, cases, edge),
            "damage": evaluate_projection(_with_lineage(graph, edges), cases),
        })

    additions = []
    for edge in admissible_false_edges(lg):
        edges = set(graph.lineage_edges)
        edges.add(edge)
        additions.append({
            "kind": "add",
            "edge": [graph.names[edge[0]], graph.names[edge[1]]],
            "edge_index": list(edge),
            "features": structural_features(graph, cases, edge),
            "damage": evaluate_projection(_with_lineage(graph, edges), cases),
        })

    return {
        "dataset": graph.dataset,
        "tables": graph.n,
        "lineage_edges": len(graph.lineage_edges),
        "frozen_cases": len(cases),
        "deletions": _summary(deletions),
        "additions": _summary(additions),
        "per_edit": {"deletions": deletions, "additions": additions},
    }


def run_paths(graph_paths: list[Path], output: Path) -> dict:
    payload = {"phase": "0A-k1", "datasets": [run_k1(load_pyg_graph(p)) for p in graph_paths]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload

