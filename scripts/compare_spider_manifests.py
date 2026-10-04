from __future__ import annotations

import argparse
import json
from pathlib import Path

from lineage_robustness.spider_manifest import compare_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare static and dbt-manifest lineage for Spider 2.0-DBT.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = compare_root(args.root)
    successful = [r for r in rows if r.parse_succeeded]
    exact = [
        r for r in successful
        if r.static_only_edges == 0 and r.manifest_only_edges == 0
    ]
    payload = {
        "summary": {
            "projects_attempted": len(rows),
            "parse_succeeded": len(successful),
            "parse_failed": len(rows) - len(successful),
            "exact_edge_match_projects": len(exact),
            "projects_with_edge_mismatch": len(successful) - len(exact),
        },
        "projects": [r.to_dict() for r in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

