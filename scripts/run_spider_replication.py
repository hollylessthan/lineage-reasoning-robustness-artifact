from __future__ import annotations

import argparse
import json
from pathlib import Path

from lineage_robustness.spider_replication import run_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exact Spider 2.0-DBT deletion-severity replication.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = run_corpus(args.root, args.output)
    compact = {
        "summary": payload["summary"],
        "deepest_projects": sorted(
            [
                {
                    "project": p["project"],
                    "depth": p["model_longest_path"],
                    "models": p["models"],
                    "edges": p["model_lineage_edges"],
                    "gini": p["gini_nonlocal_severity"],
                    "top10": p["top_10pct_share_nonlocal_severity"],
                    "max_damage": p["max_nonlocal_incorrect_elements"],
                }
                for p in payload["projects"] if p["status"] == "eligible"
            ],
            key=lambda x: (x["depth"] or -1, x["edges"]),
            reverse=True,
        )[:15],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

