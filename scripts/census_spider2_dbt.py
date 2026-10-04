from __future__ import annotations

import argparse
from pathlib import Path

from lineage_robustness.spider_census import census_spider_root, summarize, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Census Spider 2.0-DBT project lineage graphs.")
    parser.add_argument("spider_dbt_root", type=Path, help="Path to Spider2/spider2-dbt/examples")
    parser.add_argument("--output", type=Path, default=Path("results/spider2_dbt_census.json"))
    args = parser.parse_args()

    results = census_spider_root(args.spider_dbt_root)
    write_json(results, args.output)
    summary = summarize(results)

    for key, value in summary.items():
        print(f"{key}: {value}")

    if summary["projects"] == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

