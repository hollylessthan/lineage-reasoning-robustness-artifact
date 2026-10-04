from __future__ import annotations

import argparse
import json
from pathlib import Path

from lineage_robustness.phase0c import run_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 0C exact severity oracle.")
    parser.add_argument("--graph", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = run_paths(args.graph, args.output)
    compact = {
        d["dataset"]: {
            "graph_diagnostics": d["graph_diagnostics"],
            "oracle": d["oracle"],
            "deletions": d["deletions"],
            "additions": d["additions"],
        }
        for d in payload["datasets"]
    }
    print(json.dumps({
        "all_oracles_exact": payload["all_oracles_exact"],
        "datasets": compact,
    }, indent=2, sort_keys=True))
    return 0 if payload["all_oracles_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

