from __future__ import annotations


def evaluate_gate(payload: dict) -> dict:
    summary = payload["summary"]
    projects = payload["projects"]

    multi_hop_projects = sum(
        1 for p in projects
        if p.get("max_depth") is not None and p["max_depth"] >= 2
    )
    eligible_edges = sum(
        p["lineage_edges"]
        for p in projects
        if p["classification"] in {
            "lineage-only-eligible",
            "combined-impact-eligible",
        }
    )
    combined_projects = sum(
        1 for p in projects if p["classification"] == "combined-impact-eligible"
    )
    unresolved_projects = sum(1 for p in projects if p["unresolved_refs"] > 0)

    checks = {
        "at_least_10_multihop_projects": multi_hop_projects >= 10,
        "aggregate_edges_exceed_dwbench_pilot_60": eligible_edges > 60,
        "some_combined_impact_projects": combined_projects > 0,
        "corpus_nonempty": summary["projects"] > 0,
    }

    return {
        "go": all(checks.values()),
        "checks": checks,
        "metrics": {
            "projects": summary["projects"],
            "multi_hop_projects": multi_hop_projects,
            "eligible_lineage_edges": eligible_edges,
            "combined_impact_projects": combined_projects,
            "projects_with_unresolved_refs": unresolved_projects,
            "classification_counts": summary["classification_counts"],
        },
        "notes": [
            "Static census is a feasibility gate, not the final frozen corpus.",
            "Manifest-derived lineage must be compared against static extraction before final evaluation.",
            "Relationship tests are explicit FK evidence but are not assumed comprehensive.",
        ],
    }

