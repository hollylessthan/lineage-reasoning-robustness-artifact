from pathlib import Path

from lineage_robustness.spider_census import census_project_static, census_spider_root, summarize


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_census_multihop_project_with_relationships(tmp_path):
    project = tmp_path / "airbnb_like"
    write(project / "dbt_project.yml", "name: demo\n")
    write(project / "models" / "src_a.sql", "select * from {{ source('raw','a') }}")
    write(project / "models" / "mid.sql", "select * from {{ ref('src_a') }}")
    write(project / "models" / "final.sql", "select * from {{ ref('mid') }}")
    write(
        project / "models" / "schema.yml",
        "models:\n  - name: final\n    columns:\n      - name: id\n        tests:\n          - relationships:\n              to: ref('mid')\n              field: id\n",
    )

    row = census_project_static(project)
    assert row.models == 3
    assert row.sources == 1
    assert row.lineage_edges == 3
    assert row.max_depth == 3
    assert row.multi_hop_anchors >= 1
    assert row.relationship_tests == 1
    assert row.relationship_to_ref_tests == 1
    assert row.unresolved_refs == 0
    assert row.classification == "combined-impact-eligible"


def test_unresolved_ref_excludes_project(tmp_path):
    project = tmp_path / "broken"
    write(project / "dbt_project.yml", "name: demo\n")
    write(project / "models" / "final.sql", "select * from {{ ref('missing_model') }}")

    row = census_project_static(project)
    assert row.unresolved_refs == 1
    assert row.classification == "excluded"


def test_direct_only_classification(tmp_path):
    project = tmp_path / "direct"
    write(project / "dbt_project.yml", "name: demo\n")
    write(project / "models" / "a.sql", "select 1")
    write(project / "models" / "b.sql", "select * from {{ ref('a') }}")

    row = census_project_static(project)
    assert row.max_depth == 1
    assert row.classification == "direct-only-control"


def test_root_summary(tmp_path):
    for name in ("p1", "p2"):
        project = tmp_path / name
        write(project / "dbt_project.yml", "name: demo\n")
        write(project / "models" / "a.sql", "select 1")
        write(project / "models" / "b.sql", "select * from {{ ref('a') }}")

    rows = census_spider_root(tmp_path)
    summary = summarize(rows)
    assert summary["projects"] == 2
    assert summary["total_models"] == 4
    assert summary["total_lineage_edges"] == 2

