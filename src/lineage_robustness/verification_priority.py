from __future__ import annotations

import json
import math
import random
from pathlib import Path

import networkx as nx

from lineage_robustness.spider_replication import (
    load_project_model_graph,
    deletion_nonlocal_severity,
    model_depth,
    to_nx,
)


BUDGETS = (0.05, 0.10, 0.20, 0.30)



def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg
        i = j
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    if sx == 0 or sy == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def spearman(x: list[float], y: list[float]) -> float | None:
    return _pearson(_rank(x), _rank(y))


def edge_features(graph, edge: tuple[int, int]) -> dict[str, float]:
    g = to_nx(graph)
    u, v = edge
    bet = nx.edge_betweenness_centrality(g, normalized=True)
    return {
        "source_out_degree": float(g.out_degree(u)),
        "target_in_degree": float(g.in_degree(v)),
        "degree_product": float(max(1, g.out_degree(u)) * max(1, g.in_degree(v))),
        "edge_betweenness": float(bet.get(edge, 0.0)),
        "downstream_closure": float(len(nx.descendants(g, v)) + 1),
        "upstream_closure": float(len(nx.ancestors(g, u)) + 1),
    }


def expected_risk(rows: list[dict], selected: set[int]) -> float:
    return sum(
        float(row["impact"]) * float(row["error_probability"])
        for i, row in enumerate(rows)
        if i not in selected
    )


def removed_risk_fraction(rows: list[dict], selected: set[int]) -> float:
    total = expected_risk(rows, set())
    if total <= 0:
        return 0.0
    remaining = expected_risk(rows, selected)
    return (total - remaining) / total


def rank_indices(rows: list[dict], key: str) -> list[int]:
    return sorted(
        range(len(rows)),
        key=lambda i: (
            float(rows[i][key]),
            float(rows[i]["impact"]),
            rows[i]["edge"][0],
            rows[i]["edge"][1],
        ),
        reverse=True,
    )


def random_curve(rows: list[dict], budgets=BUDGETS, seeds=range(100)) -> dict[str, float]:
    n = len(rows)
    out = {}
    for budget in budgets:
        k = max(1, math.ceil(budget * n))
        vals = []
        for seed in seeds:
            idx = list(range(n))
            random.Random(seed).shuffle(idx)
            vals.append(removed_risk_fraction(rows, set(idx[:k])))
        out[f"{int(budget*100)}pct"] = sum(vals) / len(vals)
    return out


def curve(rows: list[dict], ranking: list[int], budgets=BUDGETS) -> dict[str, float]:
    n = len(rows)
    out = {}
    for budget in budgets:
        k = max(1, math.ceil(budget * n))
        out[f"{int(budget*100)}pct"] = removed_risk_fraction(rows, set(ranking[:k]))
    return out


def uniform_error_probabilities(n: int, p: float = 0.10) -> list[float]:
    return [p] * n


def heterogeneous_error_probabilities(rows: list[dict], scenario: str) -> list[float]:
    """Synthetic sensitivity scenarios.

    These are not claims about real error rates. They test whether impact-based
    prioritization remains useful when false-edge probability is correlated
    with cheap graph features.
    """
    if not rows:
        return []
    if scenario == "anti_impact_degree":
        raw = [
            1.0 / (1.0 + float(r["degree_product"]))
            for r in rows
        ]
    elif scenario == "pro_betweenness":
        raw = [
            0.25 + float(r["edge_betweenness"])
            for r in rows
        ]
    else:
        raise ValueError(scenario)

    mean = sum(raw) / len(raw)
    scale = 0.10 / mean if mean else 0.0
    return [min(0.95, max(0.001, x * scale)) for x in raw]


def make_rows(graph, error_probs: list[float]) -> list[dict]:
    base = []
    for edge in sorted(graph.edges):
        impact = deletion_nonlocal_severity(graph, edge)["nonlocal_incorrect_elements"]
        feats = edge_features(graph, edge)
        base.append({
            "edge": [graph.names[edge[0]], graph.names[edge[1]]],
            "edge_index": list(edge),
            "impact": int(impact),
            **feats,
        })
    if len(error_probs) != len(base):
        raise ValueError("probability vector length mismatch")
    for row, p in zip(base, error_probs):
        row["error_probability"] = float(p)
        row["expected_risk"] = float(p) * float(row["impact"])
    return base


def evaluate_scenario(graph, probabilities: list[float], name: str) -> dict[str, object]:
    rows = make_rows(graph, probabilities)
    rankings = {
        "impact": rank_indices(rows, "impact"),
        "expected_risk_oracle": rank_indices(rows, "expected_risk"),
        "edge_betweenness": rank_indices(rows, "edge_betweenness"),
        "degree_product": rank_indices(rows, "degree_product"),
        "downstream_closure": rank_indices(rows, "downstream_closure"),
        "upstream_closure": rank_indices(rows, "upstream_closure"),
    }
    curves = {name_: curve(rows, order) for name_, order in rankings.items()}
    curves["random"] = random_curve(rows)

    return {
        "scenario": name,
        "total_expected_risk": expected_risk(rows, set()),
        "curves": curves,
        "rows": rows,
    }


def project_result(project_dir: Path) -> dict[str, object]:
    graph = load_project_model_graph(project_dir)
    depth = model_depth(graph)
    if graph.n == 0 or graph.unresolved_refs > 0 or graph.duplicate_model_names > 0 or depth is None:
        return {
            "project": graph.project,
            "status": "excluded",
            "depth": depth,
        }
    if len(graph.edges) < 5 or depth < 2:
        return {
            "project": graph.project,
            "status": "control-only",
            "depth": depth,
        }

    n = len(graph.edges)
    scenarios = []

    uniform = uniform_error_probabilities(n)
    scenarios.append(evaluate_scenario(graph, uniform, "uniform_10pct"))

    probe_rows = make_rows(graph, uniform)
    for scenario in ("anti_impact_degree", "pro_betweenness"):
        probs = heterogeneous_error_probabilities(probe_rows, scenario)
        scenarios.append(evaluate_scenario(graph, probs, scenario))

    uniform_rows = scenarios[0]["rows"]
    impact_betweenness_spearman = spearman(
        [float(r["impact"]) for r in uniform_rows],
        [float(r["edge_betweenness"]) for r in uniform_rows],
    )

    return {
        "project": graph.project,
        "status": "eligible",
        "models": graph.n,
        "edges": n,
        "depth": depth,
        "impact_betweenness_spearman": impact_betweenness_spearman,
        "scenarios": scenarios,
    }


def macro_summary(projects: list[dict]) -> dict[str, object]:
    eligible = [p for p in projects if p["status"] == "eligible"]
    strategies = (
        "impact",
        "expected_risk_oracle",
        "edge_betweenness",
        "degree_product",
        "downstream_closure",
        "upstream_closure",
        "random",
    )

    summary: dict[str, object] = {}
    for scenario in ("uniform_10pct", "anti_impact_degree", "pro_betweenness"):
        by_strategy = {}
        for strategy in strategies:
            by_budget = {}
            for budget in BUDGETS:
                key = f"{int(budget*100)}pct"
                vals = []
                for p in eligible:
                    s = next(x for x in p["scenarios"] if x["scenario"] == scenario)
                    vals.append(float(s["curves"][strategy][key]))
                if vals:
                    vals = sorted(vals)
                    n = len(vals)
                    median = vals[n//2] if n % 2 else (vals[n//2 - 1] + vals[n//2]) / 2
                    by_budget[key] = {
                        "projects": len(vals),
                        "mean_removed_risk": sum(vals) / len(vals),
                        "median_removed_risk": median,
                        "min_removed_risk": min(vals),
                        "max_removed_risk": max(vals),
                    }
            by_strategy[strategy] = by_budget
        summary[scenario] = by_strategy
    pairwise = {}
    for scenario in ("uniform_10pct", "anti_impact_degree", "pro_betweenness"):
        per_budget = {}
        for budget in BUDGETS:
            key = f"{int(budget*100)}pct"
            deltas = []
            wins = ties = losses = 0
            for p in eligible:
                s = next(x for x in p["scenarios"] if x["scenario"] == scenario)
                delta = float(s["curves"]["impact"][key]) - float(
                    s["curves"]["edge_betweenness"][key]
                )
                deltas.append(delta)
                if delta > 1e-12:
                    wins += 1
                elif delta < -1e-12:
                    losses += 1
                else:
                    ties += 1
            per_budget[key] = {
                "impact_minus_betweenness_mean": (
                    sum(deltas) / len(deltas) if deltas else 0.0
                ),
                "impact_wins": wins,
                "ties": ties,
                "impact_losses": losses,
            }
        pairwise[scenario] = per_budget

    rank_corrs = [
        float(p["impact_betweenness_spearman"])
        for p in eligible
        if p["impact_betweenness_spearman"] is not None
    ]
    rank_corrs.sort()
    median_corr = None
    if rank_corrs:
        n = len(rank_corrs)
        median_corr = (
            rank_corrs[n // 2]
            if n % 2
            else (rank_corrs[n // 2 - 1] + rank_corrs[n // 2]) / 2
        )

    return {
        "eligible_projects": len(eligible),
        "scenarios": summary,
        "impact_vs_betweenness": {
            "projects_with_defined_spearman": len(rank_corrs),
            "median_spearman": median_corr,
            "min_spearman": min(rank_corrs) if rank_corrs else None,
            "max_spearman": max(rank_corrs) if rank_corrs else None,
            "pairwise_budget_deltas": pairwise,
        },
    }


def run_corpus(root: Path, output: Path) -> dict[str, object]:
    projects = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / "dbt_project.yml").exists():
            projects.append(project_result(child))

    payload = {
        "phase": "0E-verification-priority",
        "assumptions": {
            "uniform_primary": (
                "Primary scenario assumes equal independent 10% false-edge probability. "
                "Impact ranking is Bayes-optimal only under equal probabilities."
            ),
            "sensitivity": (
                "Two synthetic heterogeneous probability scenarios test robustness. "
                "They are stress tests, not estimates of real lineage error rates."
            ),
            "budget_interpretation": "Budget is fraction of observed lineage edges manually verified.",
        },
        "macro": macro_summary(projects),
        "projects": projects,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload

