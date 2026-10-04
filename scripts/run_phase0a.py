from __future__ import annotations

import argparse
import json
from pathlib import Path

from lineage_robustness.phase0a import run_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exhaustive k=1 lineage sensitivity.")
    parser.add_argument("--graph", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = run_paths(args.graph, args.output)
    compact = {}
    for d in payload["datasets"]:
        compact[d["dataset"]] = {
            "frozen_cases": d["frozen_cases"],
            "deletions": {k: v for k, v in d["deletions"].items() if k != "top_10"},
            "additions": {k: v for k, v in d["additions"].items() if k != "top_10"},
        }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

