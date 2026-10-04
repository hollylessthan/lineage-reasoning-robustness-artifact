from pathlib import Path

from lineage_robustness.spider_replication import (
    deletion_nonlocal_severity,
    load_project_model_graph,
    model_depth,
)


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_model_depth_excludes_sources(tmp_path):
    p = tmp_path / "proj"
    write(p / "models" / "a.sql", "select 1")
    write(p / "models" / "b.sql", "select * from {{ ref('a') }}")
    write(p / "models" / "c.sql", "select * from {{ ref('b') }}")
    graph = load_project_model_graph(p)
    assert model_depth(graph) == 2


def test_chain_middle_edge_has_nonlocal_damage(tmp_path):
    p = tmp_path / "proj"
    write(p / "models" / "a.sql", "select 1")
    write(p / "models" / "b.sql", "select * from {{ ref('a') }}")
    write(p / "models" / "c.sql", "select * from {{ ref('b') }}")
    write(p / "models" / "d.sql", "select * from {{ ref('c') }}")
    graph = load_project_model_graph(p)
    idx = {name: i for i, name in enumerate(graph.names)}
    result = deletion_nonlocal_severity(graph, (idx["b"], idx["c"]))
    assert result["nonlocal_incorrect_elements"] > 0
    assert result["alternate_path"] is False


def test_redundant_edge_zero_nonlocal_damage(tmp_path):
    p = tmp_path / "proj"
    write(p / "models" / "a.sql", "select 1")
    write(p / "models" / "b.sql", "select * from {{ ref('a') }}")
    write(p / "models" / "c.sql", "select * from {{ ref('a') }} join {{ ref('b') }}")
    graph = load_project_model_graph(p)
    idx = {name: i for i, name in enumerate(graph.names)}
    result = deletion_nonlocal_severity(graph, (idx["a"], idx["c"]))
    assert result["alternate_path"] is True
    assert result["nonlocal_incorrect_elements"] == 0


def test_duplicate_model_stems_are_flagged(tmp_path):
    p = tmp_path / "proj"
    write(p / "models" / "a" / "dup.sql", "select 1")
    write(p / "models" / "b" / "dup.sql", "select 2")
    graph = load_project_model_graph(p)
    assert graph.duplicate_model_names == 1

