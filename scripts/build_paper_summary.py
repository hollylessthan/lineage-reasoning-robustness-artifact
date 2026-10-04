from __future__ import annotations

import argparse
import json
from pathlib import Path

from lineage_robustness.spider_replication import run_corpus


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-ready Phase 0D summary table.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    payload = run_corpus(args.root, args.json)
    deep = [
        p for p in payload["projects"]
        if p["status"] == "eligible" and (p["model_longest_path"] or 0) >= 3
    ]
    deep.sort(key=lambda p: (
        -(p["model_longest_path"] or 0),
        -int(p["model_lineage_edges"]),
        str(p["project"]),
    ))

    lines = [
        "# Paper-ready Spider 2.0-DBT summary",
        "",
        "| Project | Models | Edges | Depth | Nonlocal edges | Max severity | Gini | Top-10% share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in deep:
        lines.append(
            f"| {p['project']} | {p['models']} | {p['model_lineage_edges']} | "
            f"{p['model_longest_path']} | {p['edges_with_nonzero_nonlocal_damage']} | "
            f"{p['max_nonlocal_incorrect_elements']} | {p['gini_nonlocal_severity']:.3f} | "
            f"{pct(p['top_10pct_share_nonlocal_severity'])} |"
        )

    s = payload["summary"]
    lines += [
        "",
        "## Headline results",
        "",
        f"- Eligible projects: {s['projects_eligible']} / {s['projects_total']}.",
        f"- Eligible model-to-model edges: {s['eligible_model_edges']}.",
        f"- Edges with nonzero nonlocal severity: {s['edges_with_nonzero_nonlocal_damage']}.",
        f"- Projects with model depth >=3: {s['projects_model_depth_ge_3']}.",
        f"- Projects with model depth >=4: {s['projects_model_depth_ge_4']}.",
        f"- Depth>=3 median top-10% share: {pct(s['depth_ge_3_macro']['median_top_10pct_share'])}.",
        f"- Depth>=4 median top-10% share: {pct(s['depth_ge_4_macro']['median_top_10pct_share'])}.",
        f"- Pooled top-10% share: {pct(s['pooled_top_10pct_share_nonlocal_severity'])}.",
        f"- Leave-largest-out pooled top-10% share: {pct(s['leave_largest_out_pooled_top_10pct_share'])}.",
        f"- Maximum exact nonlocal severity: {s['max_nonlocal_incorrect_elements']} incorrect answer elements.",
        "",
        "## Scope note",
        "",
        "This table is lineage-only and restricted to deterministic static-extraction eligible projects. "
        "It should not be described as representative of all Spider 2.0-DBT projects.",
        "",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

