from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re

import networkx as nx

REF_RE = re.compile(r"ref\s*\(\s*['\"]([^'\"]+)['\"](?:\s*,\s*['\"]([^'\"]+)['\"])?\s*\)")
SOURCE_RE = re.compile(r"source\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)")
REL_RE = re.compile(r"\brelationships\s*:")
REL_TO_REF_RE = re.compile(r"\bto\s*:\s*ref\s*\(")


@dataclass(frozen=True)
class ProjectCensus:
    project: str
    models: int
    sources: int
    lineage_edges: int
    max_depth: int | None
    weak_components: int
    multi_hop_anchors: int
    unresolved_refs: int
    relationship_tests: int
    relationship_to_ref_tests: int
    acyclic: bool
    extraction_mode: str
    classification: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _project_model_files(project_dir: Path) -> list[Path]:
    models = project_dir / "models"
    if not models.exists():
        return []
    return sorted(p for p in models.rglob("*.sql") if "dbt_packages" not in p.parts)


def _project_yaml_files(project_dir: Path) -> list[Path]:
    models = project_dir / "models"
    if not models.exists():
        return []
    return sorted(
        p for p in models.rglob("*")
        if p.suffix.lower() in {".yml", ".yaml"} and "dbt_packages" not in p.parts
    )


def _max_depth_if_dag(graph: nx.DiGraph) -> int | None:
    if not nx.is_directed_acyclic_graph(graph):
        return None
    return nx.algorithms.dag.dag_longest_path_length(graph)


def _multi_hop_anchor_count(graph: nx.DiGraph) -> int:
    count = 0
    for node in graph.nodes:
        lengths = nx.single_source_shortest_path_length(graph, node)
        if any(distance >= 2 for distance in lengths.values()):
            count += 1
    return count


def _classify(*, models: int, lineage_edges: int, max_depth: int | None,
              unresolved_refs: int, relationship_tests: int) -> str:
    if models == 0 or lineage_edges == 0 or unresolved_refs > 0 or max_depth is None:
        return "excluded"
    if max_depth >= 2 and relationship_tests > 0:
        return "combined-impact-eligible"
    if max_depth >= 2:
        return "lineage-only-eligible"
    return "direct-only-control"


def census_project_static(project_dir: Path) -> ProjectCensus:
    model_files = _project_model_files(project_dir)
    yaml_files = _project_yaml_files(project_dir)
    model_names = {p.stem for p in model_files}

    graph = nx.DiGraph()
    graph.add_nodes_from(model_names)
    source_nodes: set[str] = set()
    unresolved = 0

    for path in model_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        dst = path.stem

        for match in REF_RE.finditer(text):
            first, second = match.groups()
            ref_name = second or first
            if ref_name in model_names:
                graph.add_edge(ref_name, dst)
            else:
                unresolved += 1

        for match in SOURCE_RE.finditer(text):
            source = f"source:{match.group(1)}.{match.group(2)}"
            source_nodes.add(source)
            graph.add_edge(source, dst)

    yaml_text = "\n".join(
        p.read_text(encoding="utf-8", errors="replace") for p in yaml_files
    )
    relationship_tests = len(REL_RE.findall(yaml_text))
    relationship_to_ref_tests = len(REL_TO_REF_RE.findall(yaml_text))

    max_depth = _max_depth_if_dag(graph)
    weak_components = (
        nx.number_weakly_connected_components(graph)
        if graph.number_of_nodes()
        else 0
    )
    multi_hop_anchors = _multi_hop_anchor_count(graph) if max_depth is not None else 0
    classification = _classify(
        models=len(model_names),
        lineage_edges=graph.number_of_edges(),
        max_depth=max_depth,
        unresolved_refs=unresolved,
        relationship_tests=relationship_tests,
    )

    return ProjectCensus(
        project=project_dir.name,
        models=len(model_names),
        sources=len(source_nodes),
        lineage_edges=graph.number_of_edges(),
        max_depth=max_depth,
        weak_components=weak_components,
        multi_hop_anchors=multi_hop_anchors,
        unresolved_refs=unresolved,
        relationship_tests=relationship_tests,
        relationship_to_ref_tests=relationship_to_ref_tests,
        acyclic=max_depth is not None,
        extraction_mode="static",
        classification=classification,
    )


def census_spider_root(root: Path) -> list[ProjectCensus]:
    projects = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / "dbt_project.yml").exists():
            projects.append(census_project_static(child))
    return projects


def summarize(results: list[ProjectCensus]) -> dict[str, object]:
    classes: dict[str, int] = {}
    for row in results:
        classes[row.classification] = classes.get(row.classification, 0) + 1
    return {
        "projects": len(results),
        "total_models": sum(r.models for r in results),
        "total_sources": sum(r.sources for r in results),
        "total_lineage_edges": sum(r.lineage_edges for r in results),
        "multi_hop_projects": sum(1 for r in results if (r.max_depth or 0) >= 2),
        "combined_impact_projects": sum(
            1 for r in results if r.classification == "combined-impact-eligible"
        ),
        "classification_counts": classes,
    }


def write_json(results: list[ProjectCensus], output: Path) -> None:
    payload = {
        "summary": summarize(results),
        "projects": [r.to_dict() for r in results],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

