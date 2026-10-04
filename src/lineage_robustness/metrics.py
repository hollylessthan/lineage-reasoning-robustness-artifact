from __future__ import annotations

from collections.abc import Iterable
from typing import Hashable


def set_distortion(gold: Iterable[Hashable], observed: Iterable[Hashable]) -> dict[str, int | bool]:
    """Compare set-valued graph answers without redefining benchmark gold."""

    gold_set = set(gold)
    observed_set = set(observed)
    fp = observed_set - gold_set
    fn = gold_set - observed_set

    return {
        "distorted": observed_set != gold_set,
        "false_positive_elements": len(fp),
        "false_negative_elements": len(fn),
        "incorrect_elements": len(fp) + len(fn),
    }


def lineage_error_amplification(*, incorrect_answer_elements: int, incorrect_lineage_edges: int) -> float:
    """Incorrect answer elements per incorrect lineage edge."""

    if incorrect_lineage_edges < 0:
        raise ValueError("incorrect_lineage_edges must be non-negative")
    if incorrect_lineage_edges == 0:
        return 0.0 if incorrect_answer_elements == 0 else float("inf")
    return incorrect_answer_elements / incorrect_lineage_edges

