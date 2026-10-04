from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Hashable, Iterable

import networkx as nx

Node = Hashable
Edge = tuple[Node, Node]


@dataclass(frozen=True)
class ProjectionDelta:
    added: tuple[Edge, ...] = ()
    removed: tuple[Edge, ...] = ()


def _validate_count(count: int) -> None:
    if count < 0:
        raise ValueError("count must be non-negative")


def dag_levels(graph: nx.DiGraph) -> dict[Node, int]:
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("lineage graph must be a DAG")
    levels = {node: 0 for node in graph.nodes}
    for node in nx.topological_sort(graph):
        for child in graph.successors(node):
            levels[child] = max(levels[child], levels[node] + 1)
    return levels


def admissible_false_edges(graph: nx.DiGraph) -> tuple[Edge, ...]:
    """New lineage edges that respect graph layers and preserve the DAG."""
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("lineage graph must be a DAG")
    levels = dag_levels(graph)
    existing = set(graph.edges())
    candidates = []
    for u in graph.nodes:
        for v in graph.nodes:
            if u == v or (u, v) in existing:
                continue
            if levels[u] >= levels[v]:
                continue
            if nx.has_path(graph, v, u):
                continue
            candidates.append((u, v))
    return tuple(sorted(candidates, key=repr))


def add_false_lineage_edges(
    graph: nx.DiGraph,
    *,
    count: int,
    seed: int = 0,
    candidate_edges: Iterable[Edge] | None = None,
) -> tuple[nx.DiGraph, ProjectionDelta]:
    _validate_count(count)
    projected = graph.copy()
    candidates = list(admissible_false_edges(graph) if candidate_edges is None else candidate_edges)
    existing = set(graph.edges())
    valid = []
    for u, v in candidates:
        if u == v or (u, v) in existing:
            continue
        trial = graph.copy()
        trial.add_edge(u, v)
        if nx.is_directed_acyclic_graph(trial):
            valid.append((u, v))
    valid = sorted(set(valid), key=repr)
    if count > len(valid):
        raise ValueError(f"requested {count} additions but only {len(valid)} valid candidates exist")
    if count == 0:
        return projected, ProjectionDelta()
    added = tuple(sorted(random.Random(seed).sample(valid, count), key=repr))
    projected.add_edges_from(added)
    if not nx.is_directed_acyclic_graph(projected):
        raise AssertionError("projection introduced a lineage cycle")
    return projected, ProjectionDelta(added=added)


def remove_lineage_edges(
    graph: nx.DiGraph,
    *,
    count: int,
    seed: int = 0,
    protected_edges: Iterable[Edge] = (),
) -> tuple[nx.DiGraph, ProjectionDelta]:
    _validate_count(count)
    projected = graph.copy()
    protected = set(protected_edges)
    candidates = sorted((e for e in graph.edges() if e not in protected), key=repr)
    if count > len(candidates):
        raise ValueError(f"requested {count} removals but only {len(candidates)} candidates exist")
    if count == 0:
        return projected, ProjectionDelta()
    removed = tuple(sorted(random.Random(seed).sample(candidates, count), key=repr))
    projected.remove_edges_from(removed)
    return projected, ProjectionDelta(removed=removed)

