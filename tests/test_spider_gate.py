from lineage_robustness.spider_gate import evaluate_gate


def payload(projects):
    return {
        "summary": {
            "projects": len(projects),
            "classification_counts": {},
        },
        "projects": projects,
    }


def project(*, depth=2, edges=7, cls="lineage-only-eligible", unresolved=0):
    return {
        "max_depth": depth,
        "lineage_edges": edges,
        "classification": cls,
        "unresolved_refs": unresolved,
    }


def test_gate_passes_when_requirements_met():
    projects = [project() for _ in range(10)]
    projects.append(project(cls="combined-impact-eligible"))
    result = evaluate_gate(payload(projects))
    assert result["go"] is True
    assert result["metrics"]["multi_hop_projects"] == 11


def test_gate_fails_without_enough_multihop_projects():
    projects = [project() for _ in range(9)]
    projects.append(project(depth=1, cls="direct-only-control"))
    result = evaluate_gate(payload(projects))
    assert result["go"] is False
    assert result["checks"]["at_least_10_multihop_projects"] is False

