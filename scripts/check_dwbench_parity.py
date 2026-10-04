from __future__ import annotations

import argparse
import json
from pathlib import Path

from lineage_robustness.dwbench import compare_lineage_gold, load_jsonl, load_pyg_graph


def load_gold(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify pristine DW-Bench lineage semantics.")
    parser.add_argument("--graph", action="append", type=Path, required=True)
    parser.add_argument("--gold", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gold = []
    for path in args.gold:
        gold.extend(load_gold(path))

    results = [compare_lineage_gold(load_pyg_graph(path), gold) for path in args.graph]
    payload = {
        "exact": all(r["exact"] for r in results),
        "datasets": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

