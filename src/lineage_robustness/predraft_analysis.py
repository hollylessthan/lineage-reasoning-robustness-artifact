from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import networkx as nx

from lineage_robustness.spider_replication import (
    SpiderModelGraph,
    deletion_nonlocal_severity,
    gini,
    load_project_model_graph,
    model_depth,
    top_fraction_share,
    to_nx,
)
from lineage_robustness.verification_priority import spearman


REL_REF_RE = re.compile(
    r"relationships[\s\S]{0,800}?to\s*:\s*[^\n]*?ref\(\s*['\"]([^'\"]+)['\"]\s*\)",
    re.IGNORECASE,
)


def longest_path_levels(g: nx.DiGraph) -> dict[int, int]:
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("expected DAG")
    levels: dict[int, int] = {}
    for node in nx.topological_sort(g):
        preds = list(g.predecessors(node))
        levels[node] = 0 if not preds else 1 + max(levels[p] for p in preds)
    return levels


def matched_layered_null(graph: SpiderModelGraph, seed: int) -> SpiderModelGraph:
    """Sample a random DAG with the observed longest-path layer sizes.

    Every node keeps its observed longest-path layer. Each node at layer k>0
    first receives one parent drawn uniformly from layer k-1, which pins its
    longest-path layer to exactly k because all edges point from a lower to a
    higher layer. The remaining edges are sampled without replacement from
    lower-to-higher layer pairs. The null therefore preserves node count, edge
    count, longest-path depth, and the realized node count at every
    longest-path layer. It does not preserve degree sequences, sink counts, or weak-connectivity
    structure. Source counts are fixed by the preserved layer-0 nodes.
    """
    g = to_nx(graph)
    levels = longest_path_levels(g)
    depth = max(levels.values(), default=0)
    m = len(graph.edges)

    by_level: dict[int, list[int]] = {}
    for node, level in levels.items():
        by_level.setdefault(level, []).append(node)
    for level in range(depth + 1):
        if not by_level.get(level):
            raise ValueError("observed longest-path layering has an empty level")
        by_level[level].sort()

    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    for node in sorted(levels):
        level = levels[node]
        if level > 0:
            edges.add((rng.choice(by_level[level - 1]), node))

    if m < len(edges):
        raise ValueError("edge count smaller than required layer-anchoring edges")
    candidates = [
        (u, v)
        for u in range(graph.n)
        for v in range(graph.n)
        if levels[u] < levels[v] and (u, v) not in edges
    ]
    need = m - len(edges)
    if need > len(candidates):
        raise ValueError("insufficient candidate edges for matched null")
    edges.update(rng.sample(candidates, need))

    null = SpiderModelGraph(
        project=f"{graph.project}-null-{seed}",
        names=graph.names,
        edges=frozenset(edges),
        unresolved_refs=0,
        duplicate_model_names=0,
    )
    null_levels = longest_path_levels(to_nx(null))
    if null_levels != levels:
        raise AssertionError("matched null did not preserve longest-path layers")
    return null


def severity_distribution(graph: SpiderModelGraph) -> list[int]:
    return [
        int(deletion_nonlocal_severity(graph, edge)["nonlocal_incorrect_elements"])
        for edge in sorted(graph.edges)
    ]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def null_summary(graph: SpiderModelGraph, samples: int = 200) -> dict[str, object]:
    observed = severity_distribution(graph)
    observed_gini = gini(observed)
    observed_top10 = top_fraction_share(observed, 0.10)

    null_gini = []
    null_top10 = []
    for seed in range(samples):
        ng = matched_layered_null(graph, seed)
        sev = severity_distribution(ng)
        null_gini.append(gini(sev))
        null_top10.append(top_fraction_share(sev, 0.10))

    return {
        "samples": samples,
        "observed_gini": observed_gini,
        "null_gini_mean": sum(null_gini) / len(null_gini),
        "null_gini_p2_5": percentile(null_gini, 0.025),
        "null_gini_p97_5": percentile(null_gini, 0.975),
        "observed_top10_share": observed_top10,
        "null_top10_mean": sum(null_top10) / len(null_top10),
        "null_top10_p2_5": percentile(null_top10, 0.025),
        "null_top10_p97_5": percentile(null_top10, 0.975),
    }


def path_multiplicity_fraction(graph: SpiderModelGraph) -> dict[str, float | int]:
    """Fraction of reachable ordered pairs with at least two distinct paths.

    Counts are capped at 2 using DAG dynamic programming. This measures path
    multiplicity, not edge-disjointness.
    """
    g = to_nx(graph)
    topo = list(nx.topological_sort(g))
    reachable_pairs = 0
    multi_path_pairs = 0

    for source in topo:
        counts = {node: 0 for node in topo}
        counts[source] = 1
        started = False
        for u in topo:
            if u == source:
                started = True
            if not started or counts[u] == 0:
                continue
            for v in g.successors(u):
                counts[v] = min(2, counts[v] + counts[u])
        for target, count in counts.items():
            if target == source or count == 0:
                continue
            reachable_pairs += 1
            if count >= 2:
                multi_path_pairs += 1

    return {
        "reachable_pairs": reachable_pairs,
        "multi_path_pairs": multi_path_pairs,
        "multi_path_fraction": (
            multi_path_pairs / reachable_pairs if reachable_pairs else 0.0
        ),
    }


def impact_betweenness_spearman(graph: SpiderModelGraph) -> float | None:
    g = to_nx(graph)
    bet = nx.edge_betweenness_centrality(g, normalized=True)
    edges = sorted(graph.edges)
    impacts = [
        float(deletion_nonlocal_severity(graph, e)["nonlocal_incorrect_elements"])
        for e in edges
    ]
    bet_vals = [float(bet.get(e, 0.0)) for e in edges]
    return spearman(impacts, bet_vals)



def lost_reachability_pairs(graph: SpiderModelGraph, edge: tuple[int, int]) -> set[tuple[int, int]]:
    g = to_nx(graph)
    h = g.copy()
    h.remove_edge(*edge)
    lost: set[tuple[int, int]] = set()
    for s in g.nodes:
        before = nx.descendants(g, s)
        after = nx.descendants(h, s)
        for t in before - after:
            lost.add((s, t))
    return lost


def closure_nonlocal_severity(graph: SpiderModelGraph, edge: tuple[int, int]) -> int:
    """Independent endpoint-corrected severity from set-valued impact answers.

    Recomputes every downstream answer (anchors other than the edge source)
    and every upstream answer (anchors other than the edge target) from the
    transitive closures of G and G-e, and counts missing answer elements.
    This does not reuse the lost-pair enumeration used by
    ``deletion_nonlocal_severity``.
    """
    u, v = edge
    g = to_nx(graph)
    h = g.copy()
    h.remove_edge(u, v)
    tc_g = nx.transitive_closure_dag(g)
    tc_h = nx.transitive_closure_dag(h)
    downstream = sum(
        len(set(tc_g.successors(a)) - set(tc_h.successors(a)))
        for a in g.nodes
        if a != u
    )
    upstream = sum(
        len(set(tc_g.predecessors(b)) - set(tc_h.predecessors(b)))
        for b in g.nodes
        if b != v
    )
    return downstream + upstream


def lemma_diagnostics(graph: SpiderModelGraph) -> dict[str, object]:
    """Verify the reachability-loss / betweenness relationship edge by edge.

    L(e) counts reachable ordered pairs disconnected by deleting e.
    EB(e) is unnormalized directed edge betweenness.
    M(e) counts multi-path reachable pairs with at least one path through e.

    For every edge, |L(e)-EB(e)| <= M(e). For unique-path graphs M(e)=0,
    hence L(e)=EB(e). We also verify the endpoint-corrected paper severity:
    severity(e)=2*L(e)-A_u(e)-B_v(e), where A_u counts lost pairs sourced
    at edge source u and B_v counts lost pairs targeting edge target v.
    """
    g = to_nx(graph)
    bet = nx.edge_betweenness_centrality(g, normalized=False)
    rows = []
    all_bounds_hold = True
    all_endpoint_identities_hold = True
    all_independent_severity_matches = True
    for edge in sorted(graph.edges):
        u, v = edge
        lost = lost_reachability_pairs(graph, edge)
        L = len(lost)
        endpoint_source = sum(1 for s, _ in lost if s == u)
        endpoint_target = sum(1 for _, t in lost if t == v)
        exact_formula = 2 * L - endpoint_source - endpoint_target
        measured = int(
            deletion_nonlocal_severity(graph, edge)["nonlocal_incorrect_elements"]
        )

        multi_through = 0
        for s, t in nx.transitive_closure_dag(g).edges:
            paths = list(nx.all_simple_paths(g, s, t))
            if len(paths) < 2:
                continue
            if any(
                any((path[i], path[i + 1]) == edge for i in range(len(path) - 1))
                for path in paths
            ):
                multi_through += 1

        eb = float(bet.get(edge, 0.0))
        gap = eb - float(L)
        bound_holds = gap >= -1e-12 and gap <= float(multi_through) + 1e-12
        identity_holds = exact_formula == measured
        independent = closure_nonlocal_severity(graph, edge)
        independent_matches = independent == measured
        all_independent_severity_matches &= independent_matches
        all_bounds_hold &= bound_holds
        all_endpoint_identities_hold &= identity_holds
        rows.append({
            "edge": [graph.names[u], graph.names[v]],
            "lost_pairs_L": L,
            "edge_betweenness_unnormalized": eb,
            "multi_path_pairs_through_edge_M": multi_through,
            "bound_holds": bound_holds,
            "endpoint_source_pairs": endpoint_source,
            "endpoint_target_pairs": endpoint_target,
            "formula_severity": exact_formula,
            "measured_severity": measured,
            "endpoint_identity_holds": identity_holds,
            "independent_closure_severity": independent,
            "independent_severity_matches": independent_matches,
        })
    return {
        "edges": len(rows),
        "all_bounds_hold": all_bounds_hold,
        "all_endpoint_identities_hold": all_endpoint_identities_hold,
        "all_independent_severity_matches": all_independent_severity_matches,
        "rows": rows,
    }


def permutation_pvalue_spearman(
    x: list[float], y: list[float], observed: float, permutations: int = 10000
) -> float:
    rng = random.Random(20260928)
    ge = 0
    yp = list(y)
    for _ in range(permutations):
        rng.shuffle(yp)
        r = spearman(x, yp)
        if r is not None and abs(r) >= abs(observed) - 1e-15:
            ge += 1
    return (ge + 1) / (permutations + 1)


def bootstrap_spearman_ci(
    x: list[float], y: list[float], samples: int = 10000
) -> tuple[float, float]:
    rng = random.Random(20260929)
    vals: list[float] = []
    n = len(x)
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        xs = [x[i] for i in idx]
        ys = [y[i] for i in idx]
        r = spearman(xs, ys)
        if r is not None:
            vals.append(r)
    return percentile(vals, 0.025), percentile(vals, 0.975)


def leave_one_out_spearman(x: list[float], y: list[float]) -> dict[str, float | None]:
    vals = []
    for i in range(len(x)):
        xs = x[:i] + x[i + 1:]
        ys = y[:i] + y[i + 1:]
        r = spearman(xs, ys)
        if r is not None:
            vals.append(r)
    return {
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def relationship_audit(project_dir: Path, graph: SpiderModelGraph) -> dict[str, int | bool]:
    model_names = set(graph.names)
    relationship_mentions = 0
    resolvable_targets: set[str] = set()
    yaml_files = [
        p for p in project_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".yml", ".yaml"}
        and "dbt_packages" not in p.parts
    ]
    for path in yaml_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        relationship_mentions += len(re.findall(r"\brelationships\b", text, flags=re.I))
        for match in REL_REF_RE.finditer(text):
            target = match.group(1)
            if target in model_names:
                resolvable_targets.add(target)
    return {
        "yaml_files": len(yaml_files),
        "relationship_mentions": relationship_mentions,
        "resolvable_relationship_targets": len(resolvable_targets),
        "fk_capable": bool(resolvable_targets),
    }


def project_audit(project_dir: Path, graph: SpiderModelGraph) -> dict[str, object]:
    sql_files = [
        p for p in project_dir.rglob("*.sql")
        if "dbt_packages" not in p.parts
    ]
    package_files = [
        p for p in (project_dir / "packages.yml", project_dir / "dependencies.yml")
        if p.exists()
    ]
    return {
        "project": graph.project,
        "models": graph.n,
        "sql_files": len(sql_files),
        "unresolved_refs": graph.unresolved_refs,
        "has_package_file": bool(package_files),
    }


def median(values: list[float]) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def group_audit(rows: list[dict]) -> dict[str, object]:
    if not rows:
        return {"projects": 0}
    return {
        "projects": len(rows),
        "median_models": median([float(r["models"]) for r in rows]),
        "median_sql_files": median([float(r["sql_files"]) for r in rows]),
        "projects_with_package_file": sum(bool(r["has_package_file"]) for r in rows),
        "median_unresolved_refs": median([float(r["unresolved_refs"]) for r in rows]),
    }


def run(root: Path, output: Path, samples: int = 200) -> dict[str, object]:
    project_dirs = [
        p for p in sorted(root.iterdir())
        if p.is_dir() and (p / "dbt_project.yml").exists()
    ]

    projects = []
    audits = []
    relationship_rows = []
    eligible_lemma_results = []
    for pdir in project_dirs:
        graph = load_project_model_graph(pdir)
        depth = model_depth(graph)
        if graph.n == 0:
            status = "no-models"
        elif graph.duplicate_model_names > 0:
            status = "duplicate-model-names"
        elif graph.unresolved_refs > 0:
            status = "unresolved-refs"
        elif depth is None:
            status = "cycle"
        elif len(graph.edges) == 0:
            status = "no-lineage"
        else:
            status = "eligible"

        audit = project_audit(pdir, graph)
        audit["status"] = status
        audits.append(audit)

        rel = relationship_audit(pdir, graph)
        rel.update({"project": graph.project, "status": status})
        relationship_rows.append(rel)

        if status == "eligible":
            eligible_lemma_results.append({
                "project": graph.project,
                "lemma": lemma_diagnostics(graph),
            })

        if status == "eligible" and (depth or 0) >= 3:
            redundancy = path_multiplicity_fraction(graph)
            corr = impact_betweenness_spearman(graph)
            projects.append({
                "project": graph.project,
                "models": graph.n,
                "edges": len(graph.edges),
                "depth": depth,
                "redundancy": redundancy,
                "impact_betweenness_spearman": corr,
                "spearman_gap": (1.0 - corr) if corr is not None else None,
                "null": null_summary(graph, samples=samples),
                "relationship_audit": rel,
                "lemma": lemma_diagnostics(graph),
            })

    redundancy_vals = [
        float(p["redundancy"]["multi_path_fraction"])
        for p in projects if p["spearman_gap"] is not None
    ]
    gaps = [
        float(p["spearman_gap"])
        for p in projects if p["spearman_gap"] is not None
    ]

    included = [r for r in audits if r["status"] == "eligible"]
    excluded_unresolved = [r for r in audits if r["status"] == "unresolved-refs"]

    observed_corr = spearman(redundancy_vals, gaps) if gaps else None
    null_gaps = [
        float(p["null"]["null_gini_mean"]) - float(p["null"]["observed_gini"])
        for p in projects
    ]
    unified_corr = spearman(redundancy_vals, null_gaps) if redundancy_vals else None

    payload = {
        "phase": "pre-draft-gap-analysis",
        "null_model": {
            "type": "layered_random_dag",
            "preserves": [
                "node_count",
                "edge_count",
                "longest_path_depth",
                "each node's observed longest-path layer",
                "observed longest-path layer node counts",
            ],
            "does_not_preserve": [
                "in_degree_sequence",
                "out_degree_sequence",
                "sink_counts",
                "weak_connectivity",
            ],
            "samples_per_project": samples,
        },
        "depth_ge_3_projects": projects,
        "redundancy_vs_spearman_gap": {
            "projects": len(gaps),
            "spearman": observed_corr,
            "permutation_p_two_sided": (
                permutation_pvalue_spearman(redundancy_vals, gaps, observed_corr)
                if observed_corr is not None else None
            ),
            "bootstrap_95pct_ci": (
                list(bootstrap_spearman_ci(redundancy_vals, gaps))
                if observed_corr is not None else None
            ),
            "leave_one_project_out": leave_one_out_spearman(redundancy_vals, gaps),
        },
        "redundancy_vs_null_gini_gap": {
            "projects": len(projects),
            "spearman": unified_corr,
        },
        "lemma_validation": {
            "projects": len(eligible_lemma_results),
            "edges": sum(int(p["lemma"]["edges"]) for p in eligible_lemma_results),
            "all_bounds_hold": all(
                bool(p["lemma"]["all_bounds_hold"]) for p in eligible_lemma_results
            ),
            "all_endpoint_identities_hold": all(
                bool(p["lemma"]["all_endpoint_identities_hold"])
                for p in eligible_lemma_results
            ),
            "all_independent_severity_matches": all(
                bool(p["lemma"]["all_independent_severity_matches"])
                for p in eligible_lemma_results
            ),
        },
        "relationship_coverage": {
            "eligible_projects": sum(r["status"] == "eligible" for r in relationship_rows),
            "eligible_fk_capable_projects": sum(
                r["status"] == "eligible" and r["fk_capable"]
                for r in relationship_rows
            ),
            "rows": relationship_rows,
        },
        "exclusion_audit": {
            "eligible": group_audit(included),
            "excluded_unresolved_refs": group_audit(excluded_unresolved),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload

