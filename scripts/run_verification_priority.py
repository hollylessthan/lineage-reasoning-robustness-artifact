from __future__ import annotations

import argparse
import json
from pathlib import Path

from lineage_robustness.verification_priority import run_corpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = run_corpus(args.root, args.output)
    compact = {
        "assumptions": payload["assumptions"],
        "macro": payload["macro"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

