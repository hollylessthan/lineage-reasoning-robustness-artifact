from pathlib import Path

from lineage_robustness.spider_replication import load_project_model_graph
from lineage_robustness.verification_priority import (
    curve,
    expected_risk,
    make_rows,
    removed_risk_fraction,
    uniform_error_probabilities,
)


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_removed_risk_fraction():
    rows = [
        {"impact": 10, "error_probability": 0.1},
        {"impact": 1, "error_probability": 0.1},
    ]
    assert removed_risk_fraction(rows, {0}) > removed_risk_fraction(rows, {1})


def test_impact_ranking_beats_reverse_on_chain(tmp_path):
    p = tmp_path / "proj"
    write(p / "models" / "a.sql", "select 1")
    write(p / "models" / "b.sql", "select * from {{ ref('a') }}")
    write(p / "models" / "c.sql", "select * from {{ ref('b') }}")
    write(p / "models" / "d.sql", "select * from {{ ref('c') }}")
    write(p / "models" / "e.sql", "select * from {{ ref('d') }}")
    graph = load_project_model_graph(p)
    rows = make_rows(graph, uniform_error_probabilities(len(graph.edges)))
    order = sorted(range(len(rows)), key=lambda i: rows[i]["impact"], reverse=True)
    reverse = list(reversed(order))
    assert curve(rows, order, budgets=(0.25,))["25pct"] >= curve(rows, reverse, budgets=(0.25,))["25pct"]


def test_expected_risk_is_sum():
    rows = [
        {"impact": 4, "error_probability": 0.25},
        {"impact": 2, "error_probability": 0.5},
    ]
    assert expected_risk(rows, set()) == 2.0

