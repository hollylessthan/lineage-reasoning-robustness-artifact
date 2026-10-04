"""Regenerate and verify all paper evidence; no model calls or credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

from lineage_robustness.predraft_analysis import run as run_analysis
from lineage_robustness.spider_replication import run_corpus as run_spider
from lineage_robustness.verification_priority import run_corpus as run_priority, spearman
from lineage_robustness.phase0c import run_paths as run_oracle
from lineage_robustness.dwbench import compare_lineage_gold, load_pyg_graph
from check_dwbench_parity import load_gold

ROOT = Path(__file__).resolve().parents[1]
SPIDER_SHA = "cafb867313aab4e674652054198f383cf4018943"


def compare(expected, actual, path="root"):
    """Compare complete structures; counts exact, float roundoff tolerated."""
    if type(expected) is not type(actual):
        raise AssertionError(f"{path}: type mismatch")
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f"{path}: keys mismatch")
        for key in expected:
            compare(expected[key], actual[key], f"{path}/{key}")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: length mismatch")
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, f"{path}/{i}")
    elif isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-12):
            raise AssertionError(f"{path}: {expected} != {actual}")
    elif expected != actual:
        raise AssertionError(f"{path}: {expected} != {actual}")


def manuscript_checks(summary, analysis, priority):
    ps = analysis["depth_ge_3_projects"]
    def rho(rows, key="impact_betweenness_spearman"):
        return spearman([p["redundancy"]["multi_path_fraction"] for p in rows],
                        [1 - p[key] for p in rows])
    enriched = []
    for p in ps:
        q = dict(p)
        rows = p["lemma"]["rows"]
        q["lost_reachability_spearman"] = spearman(
            [r["lost_pairs_L"] for r in rows],
            [r["edge_betweenness_unnormalized"] for r in rows])
        enriched.append(q)
    gaps = [p["spearman_gap"] for p in ps]
    advantages = []
    for p in priority["projects"]:
        if p["status"] != "eligible":
            continue
        s = next(s for s in p["scenarios"] if s["scenario"] == "uniform_10pct")
        c = s["curves"]
        advantages.append(100 * (c["impact"]["10pct"] - c["edge_betweenness"]["10pct"]))
    # Independent hard gates for the paper's principal inventory and exactness claims.
    s = summary["summary"]
    assert s["projects_total"] == 69 and s["projects_eligible"] == 30
    assert s["eligible_model_edges"] == 467 and len(ps) == 10
    assert sum(p["edges"] for p in ps) == 339
    assert s["status_counts"] == {"eligible": 30, "excluded-unresolved-refs": 25,
                                  "direct-only-no-lineage": 11, "excluded-no-models": 3}
    assert analysis["lemma_validation"] == {
        "projects": 30, "edges": 467, "all_bounds_hold": True,
        "all_endpoint_identities_hold": True, "all_independent_severity_matches": True}
    return {
        "rho_primary": rho(ps),
        "rho_excluding_small_graphs": rho([p for p in ps if p["edges"] >= 6]),
        "rho_without_workday001": rho([p for p in ps if p["project"] != "workday001"]),
        "rho_without_workday002": rho([p for p in ps if p["project"] != "workday002"]),
        "rho_without_both_workday": rho([p for p in ps if not p["project"].startswith("workday")]),
        "rho_multiplicity_above_5pct": rho([p for p in ps if p["redundancy"]["multi_path_fraction"] > .05]),
        "rho_density": spearman([p["edges"] / p["models"] for p in ps], gaps),
        "rho_lost_reachability": rho(enriched, "lost_reachability_spearman"),
        "rho_lost_reachability_excluding_small": rho([p for p in enriched if p["edges"] >= 6], "lost_reachability_spearman"),
        "null_above_mean": sum(p["null"]["observed_gini"] > p["null"]["null_gini_mean"] for p in ps),
        "null_above_p97_5": sum(p["null"]["observed_gini"] > p["null"]["null_gini_p97_5"] for p in ps),
        "null_below_p2_5": sum(p["null"]["observed_gini"] < p["null"]["null_gini_p2_5"] for p in ps),
        "median_null_gini": statistics.median(p["null"]["null_gini_mean"] for p in ps),
        "median_null_top10_share": statistics.median(p["null"]["null_top10_mean"] for p in ps),
        "verification_projects": len(advantages),
        "verification_ties": sum(abs(x) < 1e-10 for x in advantages),
        "verification_exact_wins": sum(x > 1e-10 for x in advantages),
        "verification_mean_advantage_pp": statistics.mean(advantages),
        "verification_max_advantage_pp": max(advantages),
        "verification_median_advantage_pp": statistics.median(advantages),
    }


def verify(name, actual):
    expected = json.loads((ROOT / "expected" / name).read_text())
    compare(expected, actual, name)
    print(f"PASS {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spider-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--dwbench-dir", type=Path, default=ROOT / ".cache/dwbench")
    parser.add_argument("--skip-dwbench", action="store_true", help="Partial Spider-only verification")
    args = parser.parse_args()
    sha = subprocess.check_output(["git", "-C", str(args.spider_root), "rev-parse", "HEAD"], text=True).strip()
    if sha != SPIDER_SHA:
        raise SystemExit(f"Wrong Spider revision: {sha}")
    if subprocess.check_output(["git", "-C", str(args.spider_root), "status", "--porcelain", "--", "."], text=True).strip():
        raise SystemExit("Spider input tree has local changes")
    manifest = json.loads((ROOT / "provenance.json").read_text())
    for name, digest in manifest["expected_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Frozen evidence checksum mismatch: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = run_spider(args.spider_root, args.output / "paper_spider_summary.json")
    verify("paper_spider_summary.json", summary)
    analysis = run_analysis(args.spider_root, args.output / "predraft_gap_analysis.json", samples=200)
    verify("predraft_gap_analysis.json", analysis)
    priority = run_priority(args.spider_root, args.output / "verification_priority.json")
    verify("verification_priority.json", priority)
    checks = manuscript_checks(summary, analysis, priority)
    (args.output / "manuscript_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    verify("manuscript_checks.json", checks)
    if not args.skip_dwbench:
        metadata = args.output / "dwbench_asset_revision.json"
        subprocess.run([sys.executable, str(ROOT / "scripts/fetch_dwbench_assets.py"),
                        "--output-dir", str(args.dwbench_dir), "--metadata", str(metadata)], check=True)
        meta = json.loads(metadata.read_text())
        files = meta["files"]
        paths = [Path(files[k]) for k in ("tpc_di_graph", "adventureworks_graph")]
        gold_paths = [Path(files["tier1"])] if meta["qa_mode"] == "combined_jsonl" else [Path(files[k]) for k in ("tpc_di_qa", "adventureworks_qa")]
        gold = [r for path in gold_paths for r in load_gold(path)]
        parity_rows = [compare_lineage_gold(load_pyg_graph(p), gold) for p in paths]
        parity = {"exact": all(r["exact"] for r in parity_rows), "datasets": parity_rows}
        (args.output / "dwbench_pristine_parity.json").write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n")
        verify("dwbench_pristine_parity.json", parity)
        oracle = run_oracle(paths, args.output / "phase0c_exact_severity_oracle.json")
        verify("phase0c_exact_severity_oracle.json", oracle)
        assert oracle["all_oracles_exact"]
        assert sum(d["oracle"]["edits_checked"] for d in oracle["datasets"]) == 1848
        assert all(d["oracle"]["mismatches"] == 0 for d in oracle["datasets"])
    import networkx
    report = {"status": "PARTIAL_PASS" if args.skip_dwbench else "PASS", "python": sys.version,
              "networkx": networkx.__version__, "spider_sha": sha, "dwbench_checked": not args.skip_dwbench}
    (args.output / "reproduction_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
