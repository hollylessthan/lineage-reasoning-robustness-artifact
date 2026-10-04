from __future__ import annotations

import argparse
import json
from pathlib import Path

from lineage_robustness.phase0b import run_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 0B anchor-complete k=1 analysis.")
    parser.add_argument("--graph", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = run_paths(args.graph, args.output)
    compact = {
        d["dataset"]: {
            "tables": d["tables"],
            "probes": d["probes"],
            "probe_inventory": d["probe_inventory"],
            "deletions": d["deletions"],
            "additions": d["additions"],
        }
        for d in payload["datasets"]
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

