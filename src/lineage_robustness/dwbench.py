from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class DWGraph:
    dataset: str
    names: tuple[str, ...]
    fk_edges: frozenset[tuple[int, int]]
    lineage_edges: frozenset[tuple[int, int]]

    @property
    def n(self) -> int:
        return len(self.names)


def load_pyg_graph(path: Path) -> DWGraph:
    import torch

    data = torch.load(path, weights_only=False)
    names = tuple(data["table"].table_names)

    def edge_set(edge_type: tuple[str, str, str]) -> frozenset[tuple[int, int]]:
        if edge_type not in data.edge_types:
            return frozenset()
        edge_index = data[edge_type].edge_index
        return frozenset(
            (int(edge_index[0, i]), int(edge_index[1, i]))
            for i in range(edge_index.shape[1])
        )

    return DWGraph(
        dataset=str(data.dataset_name),
        names=names,
        fk_edges=edge_set(("table", "fk_to", "table")),
        lineage_edges=edge_set(("table", "derived_from", "table")),
    )


def _adj(n: int, edges: frozenset[tuple[int, int]]) -> dict[int, set[int]]:
    out = {i: set() for i in range(n)}
    for src, dst in edges:
        out[src].add(dst)
    return out


def _reverse(n: int, edges: frozenset[tuple[int, int]]) -> dict[int, set[int]]:
    out = {i: set() for i in range(n)}
    for src, dst in edges:
        out[dst].add(src)
    return out


def trace_forward(adj: dict[int, set[int]], source: int) -> set[int]:
    visited: set[int] = set()
    stack = [source]
    reachable: set[int] = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for target in adj.get(node, set()):
            if target not in visited:
                reachable.add(target)
                stack.append(target)
    return reachable


def generate_lineage_questions(graph: DWGraph) -> list[dict]:
    names = graph.names
    dataset = graph.dataset
    lineage_adj = _adj(graph.n, graph.lineage_edges)
    lineage_adj_rev = _reverse(graph.n, graph.lineage_edges)
    fk_adj_rev = _reverse(graph.n, graph.fk_edges)

    questions: list[dict] = []
    source_to_targets = {
        src: sorted(targets)
        for src, targets in lineage_adj.items()
        if targets
    }
    target_to_sources = {
        tgt: sorted(sources)
        for tgt, sources in lineage_adj_rev.items()
        if sources
    }

    for src_idx, target_idxs in source_to_targets.items():
        target_names = sorted(names[t] for t in target_idxs)
        questions.append({
            "id": f"{dataset}_lineage_fwd_{len(questions):03d}",
            "dataset": dataset,
            "type": "lineage_impact",
            "subtype": "forward",
            "question": (
                "Which tables in the data warehouse are directly "
                f"derived from {names[src_idx]}?"
            ),
            "answer": target_names,
            "answer_type": "list",
            "difficulty": "easy",
        })

    for tgt_idx, src_idxs in target_to_sources.items():
        src_names = sorted(names[s] for s in src_idxs)
        questions.append({
            "id": f"{dataset}_lineage_rev_{len(questions):03d}",
            "dataset": dataset,
            "type": "lineage_impact",
            "subtype": "reverse",
            "question": (
                f"What are all the source tables that {names[tgt_idx]} "
                "is derived from?"
            ),
            "answer": src_names,
            "answer_type": "list",
            "difficulty": "easy",
        })

    for src_idx in source_to_targets:
        direct = lineage_adj[src_idx]
        transitive = trace_forward(lineage_adj, src_idx)
        indirect = transitive - direct - {src_idx}
        if indirect:
            questions.append({
                "id": f"{dataset}_lineage_trans_{len(questions):03d}",
                "dataset": dataset,
                "type": "lineage_impact",
                "subtype": "transitive",
                "question": (
                    f"If the schema of {names[src_idx]} changes, which "
                    "tables are directly or indirectly affected through data lineage?"
                ),
                "answer": sorted(names[t] for t in transitive),
                "answer_type": "list",
                "difficulty": "hard",
            })

    for src_idx in source_to_targets:
        lineage_reached = trace_forward(lineage_adj, src_idx)
        nodes_to_check = lineage_reached | {src_idx}
        fk_affected: set[int] = set()
        for node in nodes_to_check:
            for child in fk_adj_rev.get(node, set()):
                if child not in nodes_to_check:
                    fk_affected.add(child)
        if fk_affected:
            questions.append({
                "id": f"{dataset}_lineage_combined_{len(questions):03d}",
                "dataset": dataset,
                "type": "lineage_impact",
                "subtype": "combined_impact",
                "question": (
                    f"If {names[src_idx]} changes, which tables are affected "
                    "either through direct/indirect data lineage OR because they "
                    "have a foreign key dependency on an affected table?"
                ),
                "answer": sorted(names[t] for t in (lineage_reached | fk_affected)),
                "answer_type": "list",
                "difficulty": "hard",
            })

    for tgt_idx, src_idxs in target_to_sources.items():
        if len(src_idxs) >= 3:
            questions.append({
                "id": f"{dataset}_lineage_multisrc_{len(questions):03d}",
                "dataset": dataset,
                "type": "lineage_impact",
                "subtype": "multi_source",
                "question": (
                    f"{names[tgt_idx]} integrates data from multiple sources. "
                    f"List ALL source tables that feed into {names[tgt_idx]}."
                ),
                "answer": sorted(names[s] for s in src_idxs),
                "answer_type": "list",
                "difficulty": "hard",
            })

    return questions


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalize_answer(answer):
    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                answer = json.loads(stripped)
            except json.JSONDecodeError:
                pass
    if isinstance(answer, list):
        return sorted(str(x) for x in answer)
    return answer


def lineage_signature(row: dict) -> tuple:
    return (
        row.get("dataset"),
        row.get("subtype"),
        row.get("question"),
        json.dumps(_normalize_answer(row.get("answer")), sort_keys=True),
    )


def compare_lineage_gold(
    graph: DWGraph,
    gold_rows: list[dict],
) -> dict[str, object]:
    supported = {"forward", "reverse", "transitive", "combined_impact", "multi_source"}
    gold = [
        row for row in gold_rows
        if row.get("dataset") == graph.dataset
        and row.get("subtype") in supported
    ]
    generated = generate_lineage_questions(graph)

    gold_set = {lineage_signature(row) for row in gold}
    generated_set = {lineage_signature(row) for row in generated}

    by_subtype: dict[str, dict[str, int]] = {}
    for subtype in sorted(supported):
        g = {x for x in gold_set if x[1] == subtype}
        p = {x for x in generated_set if x[1] == subtype}
        by_subtype[subtype] = {
            "gold": len(g),
            "generated": len(p),
            "missing": len(g - p),
            "extra": len(p - g),
        }

    return {
        "dataset": graph.dataset,
        "gold_lineage_questions": len(gold_set),
        "generated_lineage_questions": len(generated_set),
        "missing": len(gold_set - generated_set),
        "extra": len(generated_set - gold_set),
        "exact": gold_set == generated_set,
        "by_subtype": by_subtype,
        "missing_examples": [list(x) for x in sorted(gold_set - generated_set)[:10]],
        "extra_examples": [list(x) for x in sorted(generated_set - gold_set)[:10]],
    }

