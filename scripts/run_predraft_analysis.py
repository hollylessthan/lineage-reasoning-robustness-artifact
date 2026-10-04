from __future__ import annotations

import argparse
import json
from pathlib import Path

from lineage_robustness.predraft_analysis import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded pre-draft gap analysis.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    payload = run(args.root, args.output, samples=args.samples)
    compact = {
        "null_model": payload["null_model"],
        "depth_ge_3_projects": payload["depth_ge_3_projects"],
        "redundancy_vs_spearman_gap": payload["redundancy_vs_spearman_gap"],
        "relationship_coverage": {
            "eligible_projects": payload["relationship_coverage"]["eligible_projects"],
            "eligible_fk_capable_projects": payload["relationship_coverage"]["eligible_fk_capable_projects"],
        },
        "exclusion_audit": payload["exclusion_audit"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

