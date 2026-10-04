from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import networkx as nx

from lineage_robustness.spider_census import REF_RE, _project_model_files


@dataclass(frozen=True)
class SpiderModelGraph:
    project: str
    names: tuple[str, ...]
    edges: frozenset[tuple[int, int]]
    unresolved_refs: int
    duplicate_model_names: int

    @property
    def n(self) -> int:
        return len(self.names)


def load_project_model_graph(project_dir: Path) -> SpiderModelGraph:
    model_files = _project_model_files(project_dir)
    stems = [p.stem for p in model_files]
    duplicate_model_names = len(stems) - len(set(stems))
    names = tuple(sorted(set(stems)))
    index = {name: i for i, name in enumerate(names)}
    edges: set[tuple[int, int]] = set()
    unresolved = 0

    for path in model_files:
        dst = path.stem
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in REF_RE.finditer(text):
            first, second = match.groups()
            ref_name = second or first
            if ref_name in index:
                edges.add((index[ref_name], index[dst]))
            else:
                unresolved += 1

    return SpiderModelGraph(
        project=project_dir.name,
        names=names,
        edges=frozenset(edges),
        unresolved_refs=unresolved,
        duplicate_model_names=duplicate_model_names,
    )


def to_nx(graph: SpiderModelGraph) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(graph.n))
    g.add_edges_from(graph.edges)
    return g


def topology_hash(graph: SpiderModelGraph) -> str:
    g = to_nx(graph)
    return nx.weisfeiler_lehman_graph_hash(g, iterations=5, digest_size=16)


def model_depth(graph: SpiderModelGraph) -> int | None:
    g = to_nx(graph)
    if not nx.is_directed_acyclic_graph(g):
        return None
    return nx.algorithms.dag.dag_longest_path_length(g)


def deletion_nonlocal_severity(
    graph: SpiderModelGraph,
    edge: tuple[int, int],
) -> dict[str, int | bool]:
    """Exact nonlocal reachability damage from deleting one observed edge.

    Direct endpoint effects are excluded:
    - downstream probe anchored at edge source;
    - upstream probe anchored at edge target.

    Every lost reachability pair contributes once to a downstream answer and
    once to an upstream answer, except those endpoint-anchored cases.
    """
    u, v = edge
    pristine = to_nx(graph)
    projected = pristine.copy()
    projected.remove_edge(u, v)

    alternate_path = nx.has_path(projected, u, v)

    lost_pairs: list[tuple[int, int]] = []
    for a in pristine.nodes:
        before = nx.descendants(pristine, a)
        after = nx.descendants(projected, a)
        for b in before - after:
            lost_pairs.append((a, b))

    downstream_nonlocal = sum(1 for a, _ in lost_pairs if a != u)
    upstream_nonlocal = sum(1 for _, b in lost_pairs if b != v)
    incorrect_elements = downstream_nonlocal + upstream_nonlocal

    return {
        "alternate_path": alternate_path,
        "lost_reachability_pairs": len(lost_pairs),
        "downstream_nonlocal_incorrect_elements": downstream_nonlocal,
        "upstream_nonlocal_incorrect_elements": upstream_nonlocal,
        "nonlocal_incorrect_elements": incorrect_elements,
    }


def gini(values: list[float]) -> float:
    vals = sorted(max(0.0, float(v)) for v in values)
    total = sum(vals)
    if not vals or total == 0:
        return 0.0
    n = len(vals)
    weighted = sum((i + 1) * v for i, v in enumerate(vals))
    return (2 * weighted) / (n * total) - (n + 1) / n


def top_fraction_share(values: list[float], fraction: float = 0.10) -> float:
    vals = sorted((max(0.0, float(v)) for v in values), reverse=True)
    total = sum(vals)
    if not vals or total == 0:
        return 0.0
    k = max(1, math.ceil(fraction * len(vals)))
    return sum(vals[:k]) / total


def project_result(project_dir: Path) -> dict[str, object]:
    graph = load_project_model_graph(project_dir)
    g = to_nx(graph)
    depth = model_depth(graph)

    if graph.n == 0:
        status = "excluded-no-models"
    elif graph.duplicate_model_names > 0:
        status = "excluded-duplicate-model-names"
    elif graph.unresolved_refs > 0:
        status = "excluded-unresolved-refs"
    elif depth is None:
        status = "excluded-cycle"
    elif len(graph.edges) == 0:
        status = "direct-only-no-lineage"
    else:
        status = "eligible"

    rows = []
    if status == "eligible":
        for edge in sorted(graph.edges):
            sev = deletion_nonlocal_severity(graph, edge)
            rows.append({
                "edge_index": list(edge),
                "edge": [graph.names[edge[0]], graph.names[edge[1]]],
                **sev,
            })

    severities = [int(r["nonlocal_incorrect_elements"]) for r in rows]
    return {
        "project": graph.project,
        "status": status,
        "models": graph.n,
        "model_lineage_edges": len(graph.edges),
        "model_longest_path": depth,
        "unresolved_refs": graph.unresolved_refs,
        "duplicate_model_names": graph.duplicate_model_names,
        "topology_hash": topology_hash(graph) if graph.n else None,
        "edges_with_nonzero_nonlocal_damage": sum(v > 0 for v in severities),
        "max_nonlocal_incorrect_elements": max(severities, default=0),
        "mean_nonlocal_incorrect_elements": (
            sum(severities) / len(severities) if severities else 0.0
        ),
        "gini_nonlocal_severity": gini(severities),
        "top_10pct_share_nonlocal_severity": top_fraction_share(severities, 0.10),
        "per_edge": rows,
    }


def run_corpus(root: Path, output: Path) -> dict[str, object]:
    projects = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / "dbt_project.yml").exists():
            projects.append(project_result(child))

    eligible = [p for p in projects if p["status"] == "eligible"]
    depth4 = [p for p in eligible if (p["model_longest_path"] or 0) >= 4]
    depth3 = [p for p in eligible if (p["model_longest_path"] or 0) >= 3]
    all_edges = [
        int(row["nonlocal_incorrect_elements"])
        for p in eligible
        for row in p["per_edge"]
    ]

    status_counts: dict[str, int] = {}
    for p in projects:
        status_counts[p["status"]] = status_counts.get(p["status"], 0) + 1

    topology_groups: dict[str, list[str]] = {}
    for p in eligible:
        topology_groups.setdefault(str(p["topology_hash"]), []).append(str(p["project"]))
    duplicate_topology_groups = {
        h: names for h, names in topology_groups.items() if len(names) > 1
    }

    def median(values: list[float]) -> float | None:
        if not values:
            return None
        vals = sorted(values)
        n = len(vals)
        if n % 2:
            return vals[n // 2]
        return (vals[n // 2 - 1] + vals[n // 2]) / 2.0

    def macro(rows: list[dict]) -> dict[str, float | int | None]:
        top10 = [float(p["top_10pct_share_nonlocal_severity"]) for p in rows]
        ginis = [float(p["gini_nonlocal_severity"]) for p in rows]
        return {
            "projects": len(rows),
            "median_top_10pct_share": median(top10),
            "min_top_10pct_share": min(top10) if top10 else None,
            "max_top_10pct_share": max(top10) if top10 else None,
            "median_gini": median(ginis),
        }

    largest = max(
        eligible,
        key=lambda p: int(p["model_lineage_edges"]),
        default=None,
    )
    without_largest = [p for p in eligible if p is not largest]
    edges_without_largest = [
        int(row["nonlocal_incorrect_elements"])
        for p in without_largest
        for row in p["per_edge"]
    ]

    payload = {
        "phase": "0D-spider-lineage-only-exact-replication",
        "summary": {
            "projects_total": len(projects),
            "status_counts": status_counts,
            "projects_eligible": len(eligible),
            "unique_eligible_topologies": len(topology_groups),
            "duplicate_topology_groups": duplicate_topology_groups,
            "projects_model_depth_ge_3": len(depth3),
            "projects_model_depth_ge_4": len(depth4),
            "eligible_model_edges": sum(int(p["model_lineage_edges"]) for p in eligible),
            "edges_with_nonzero_nonlocal_damage": sum(
                int(p["edges_with_nonzero_nonlocal_damage"]) for p in eligible
            ),
            "pooled_gini_nonlocal_severity": gini(all_edges),
            "pooled_top_10pct_share_nonlocal_severity": top_fraction_share(all_edges, 0.10),
            "depth_ge_3_macro": macro(depth3),
            "depth_ge_4_macro": macro(depth4),
            "largest_project": (
                {
                    "project": largest["project"],
                    "edges": largest["model_lineage_edges"],
                }
                if largest else None
            ),
            "leave_largest_out_pooled_gini": gini(edges_without_largest),
            "leave_largest_out_pooled_top_10pct_share": top_fraction_share(
                edges_without_largest, 0.10
            ),
            "max_nonlocal_incorrect_elements": max(all_edges, default=0),
        },
        "projects": projects,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload

