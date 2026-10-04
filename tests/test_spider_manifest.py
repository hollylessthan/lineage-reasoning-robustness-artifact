import json
from pathlib import Path

from lineage_robustness.spider_manifest import manifest_edges, static_edges_from_project


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_static_edges_match_expected(tmp_path):
    project = tmp_path / "p"
    write(project / "models" / "a.sql", "select * from {{ source('raw','a') }}")
    write(project / "models" / "b.sql", "select * from {{ ref('a') }}")
    assert static_edges_from_project(project) == {
        ("source:raw.a", "model:a"),
        ("model:a", "model:b"),
    }


def test_manifest_edges_filters_to_project_models(tmp_path):
    manifest = {
        "metadata": {"project_name": "demo"},
        "nodes": {
            "model.demo.a": {
                "resource_type": "model",
                "package_name": "demo",
                "name": "a",
                "depends_on": {"nodes": ["source.demo.raw_a"]},
            },
            "model.demo.b": {
                "resource_type": "model",
                "package_name": "demo",
                "name": "b",
                "depends_on": {"nodes": ["model.demo.a"]},
            },
            "model.pkg.x": {
                "resource_type": "model",
                "package_name": "pkg",
                "name": "x",
                "depends_on": {"nodes": []},
            },
        },
        "sources": {
            "source.demo.raw_a": {
                "source_name": "raw",
                "name": "a",
            }
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    models, sources, edges = manifest_edges(path)
    assert models == {"model:a", "model:b"}
    assert sources == {"source:raw.a"}
    assert edges == {
        ("source:raw.a", "model:a"),
        ("model:a", "model:b"),
    }

