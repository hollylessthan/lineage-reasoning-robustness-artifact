from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import subprocess


@dataclass(frozen=True)
class ManifestComparison:
    project: str
    parse_succeeded: bool
    manifest_models: int
    manifest_sources: int
    manifest_edges: int
    static_edges: int
    shared_edges: int
    static_only_edges: int
    manifest_only_edges: int
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _edge_key(unique_id: str, nodes: dict, sources: dict) -> str | None:
    if unique_id in nodes:
        node = nodes[unique_id]
        if node.get("resource_type") != "model":
            return None
        return f"model:{node.get('name')}"
    if unique_id in sources:
        source = sources[unique_id]
        return f"source:{source.get('source_name')}.{source.get('name')}"
    return None


def manifest_edges(manifest_path: Path) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodes = payload.get("nodes", {})
    sources = payload.get("sources", {})
    project_name = payload.get("metadata", {}).get("project_name")

    models = {
        uid: node
        for uid, node in nodes.items()
        if node.get("resource_type") == "model"
        and (project_name is None or node.get("package_name") == project_name)
    }

    model_names = {node.get("name") for node in models.values()}
    source_names: set[str] = set()
    edges: set[tuple[str, str]] = set()

    for uid, node in models.items():
        dst = f"model:{node.get('name')}"
        for dep in node.get("depends_on", {}).get("nodes", []):
            if dep in sources:
                src_info = sources[dep]
                src = f"source:{src_info.get('source_name')}.{src_info.get('name')}"
                source_names.add(src)
                edges.add((src, dst))
            elif dep in models:
                src = f"model:{models[dep].get('name')}"
                edges.add((src, dst))
            elif dep in nodes:
                dep_node = nodes[dep]
                if (
                    dep_node.get("resource_type") == "model"
                    and dep_node.get("name") in model_names
                ):
                    edges.add((f"model:{dep_node.get('name')}", dst))

    return {f"model:{name}" for name in model_names}, source_names, edges


def static_edges_from_project(project_dir: Path) -> set[tuple[str, str]]:
    from lineage_robustness.spider_census import REF_RE, SOURCE_RE, _project_model_files

    model_files = _project_model_files(project_dir)
    model_names = {p.stem for p in model_files}
    edges: set[tuple[str, str]] = set()

    for path in model_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        dst = f"model:{path.stem}"

        for match in REF_RE.finditer(text):
            first, second = match.groups()
            ref_name = second or first
            if ref_name in model_names:
                edges.add((f"model:{ref_name}", dst))

        for match in SOURCE_RE.finditer(text):
            edges.add((f"source:{match.group(1)}.{match.group(2)}", dst))

    return edges


def run_dbt_parse(project_dir: Path, *, timeout_seconds: int = 180) -> Path:
    project_dir = project_dir.resolve()
    target_dir = (project_dir / "target_lrr").resolve()
    cmd = [
        "dbt",
        "parse",
        "--project-dir",
        str(project_dir),
        "--profiles-dir",
        str(project_dir),
        "--target-path",
        str(target_dir),
        "--no-partial-parse",
        "--quiet",
    ]
    completed = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if len(detail) > 2000:
            detail = detail[-2000:]
        raise RuntimeError(
            f"dbt parse failed with exit {completed.returncode}: {detail}"
        )
    manifest = target_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"dbt parse succeeded but no manifest at {manifest}")
    return manifest


def compare_project(project_dir: Path) -> ManifestComparison:
    static = static_edges_from_project(project_dir)
    try:
        manifest_path = run_dbt_parse(project_dir)
        models, sources, manifest = manifest_edges(manifest_path)
        return ManifestComparison(
            project=project_dir.name,
            parse_succeeded=True,
            manifest_models=len(models),
            manifest_sources=len(sources),
            manifest_edges=len(manifest),
            static_edges=len(static),
            shared_edges=len(static & manifest),
            static_only_edges=len(static - manifest),
            manifest_only_edges=len(manifest - static),
        )
    except Exception as exc:
        return ManifestComparison(
            project=project_dir.name,
            parse_succeeded=False,
            manifest_models=0,
            manifest_sources=0,
            manifest_edges=0,
            static_edges=len(static),
            shared_edges=0,
            static_only_edges=len(static),
            manifest_only_edges=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def compare_root(root: Path) -> list[ManifestComparison]:
    out = []
    for project_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if (project_dir / "dbt_project.yml").exists() and (project_dir / "profiles.yml").exists():
            out.append(compare_project(project_dir))
    return out

