# Missing Lineage Edges in Analytics DAGs

Reproduction artifact for **Missing Lineage Edges in Analytics DAGs: Severity, Path Redundancy, and Betweenness**, prepared for IEEE ICSC 2027. This repository provides structural lineage analysis, exact severity validation, frozen evidence, and a command that regenerates and checks the reported results. It uses no LLM calls, paid API, or database credentials.

The validated reproduction artifact is published as [artifact-v1.0](https://github.com/hollylessthan/lineage-reasoning-robustness-artifact/releases/tag/artifact-v1.0) and archived on Zenodo with DOI [10.5281/zenodo.23131405](https://doi.org/10.5281/zenodo.23131405). The released commit passed all 43 unit tests and full paper reproduction in GitHub CI. This release makes the research artifact available; it does not imply paper acceptance or a production deployment.

## Reproduce the paper

Use Python 3.11 on Linux. The reference runs used Python 3.11.16 and NetworkX 3.6.1; PyTorch is needed only to load DW-Bench graph files. Network access is needed to install packages and fetch the pinned upstream inputs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0'
python -m pip install 'torch-geometric==2.8.0.post1'
python -m pytest -q
git clone --filter=blob:none --no-checkout https://github.com/xlang-ai/Spider2.git inputs/Spider2
git -C inputs/Spider2 sparse-checkout set spider2-dbt/examples
git -C inputs/Spider2 checkout cafb867313aab4e674652054198f383cf4018943
python scripts/reproduce_paper.py --spider-root inputs/Spider2/spider2-dbt/examples
```

The command regenerates six JSON outputs, checks the complete structures against `expected/` (integer and categorical fields exactly, floats within roundoff tolerance), and writes `results/reproduction_report.json`. A full successful run ends in `PASS`. It checks the Spider input revision, rejects modified corpus files, and checks frozen evidence hashes. DW-Bench v1.0.0 is fetched and verified against its recorded SHA256 before loading; gold questions are not regenerated after graph edits.

For a lighter run that needs only NetworkX, omit the PyTorch installations and add `--skip-dwbench`. That run ends in `PARTIAL_PASS` and explicitly records that DW-Bench was not checked. It is not the full release gate.

## Expected evidence

| Check | Expected result |
|---|---|
| Spider projects | 69 inspected: 30 eligible, 25 unresolved-reference, 11 no-model-edge, 3 no-model-SQL |
| Eligible edges | 467; independent closure severity matches on every edge |
| Deeper-project sample | 10 projects, 339 edges; model-only depth at least 3 |
| DW-Bench pristine parity | 47 TPC-DI and 76 AdventureWorks items, zero differences |
| DW-Bench exact oracle | 336 + 1,512 = 1,848 single-edge edits; zero mismatches |
| Multiplicity versus severity-ranking gap | Spearman 0.9515; permutation p < 0.001 |
| Small-graph exclusion | Spearman 0.9286 |
| Exclude both Workday graphs | Spearman 0.9048 |
| Matched nulls | 200 per deeper project; 3 above mean, 0 above 97.5th percentile, 3 below 2.5th percentile |
| Verification at 10% budget | Exact severity and betweenness tie on 16/17 projects; mean advantage 0.1824 percentage points |

`expected/manuscript_checks.json` records additional manuscript sensitivities. The per-project and per-edge files contain the numbers for Table I and the redundancy scatter plot. `provenance.json` records source commits, scientific workflow runs, upstream revision, package versions, and SHA256 hashes. The corrected layer-preserving null output is the reference, not the superseded null model.

## Figure 1 data

Figure 1 uses the 10 records in `predraft_gap_analysis.json` under `depth_ge_3_projects`. For each record, the label comes from `project`, the x coordinate is `100 * redundancy.multi_path_fraction`, and the y coordinate is `spearman_gap`, equivalently `1 - impact_betweenness_spearman`. Use `expected/predraft_gap_analysis.json` for the frozen reference or `results/predraft_gap_analysis.json` after reproduction. The manuscript rounds x to two decimal places and y to four for its plotted coordinates. The two Workday points have a combined label but remain separate observations. All values come from the static lineage analysis; there are no model-generated or hand-labeled plot values.

## Expected NetworkX warnings

NetworkX can print two UserWarnings about graph hashes changing in version 3.5: one for directed graphs and one for graphs without node or edge attributes. These are compatibility notices from Weisfeiler-Lehman topology hashing, not failed severity checks. Both the frozen reference and this artifact use the pinned NetworkX 3.6.1 implementation. The reproduction command compares the recorded hashes and every scientific output against the frozen evidence and exits on a difference. Warnings remain visible so unexpected warnings are not hidden; the final PASS/PARTIAL_PASS status and exit code report verification success.

## Permanent archive

The released `artifact-v1.0` snapshot is archived at [https://doi.org/10.5281/zenodo.23131405](https://doi.org/10.5281/zenodo.23131405). This version-specific DOI identifies the exact code and frozen evidence used for the paper. The corresponding [GitHub release](https://github.com/hollylessthan/lineage-reasoning-robustness-artifact/releases/tag/artifact-v1.0) provides the same tagged source snapshot. Cite the DOI for reproducibility and use the repository for browsing and subsequent development. Documentation updates on `main` do not change the released snapshot.

## Contents and scope

- `src/lineage_robustness/`: static dbt extraction, reachability severity, betweenness, capped path counting, matched DAG nulls, and independent oracle implementations.
- `scripts/`: reproduction entry point plus individual analysis commands. Earlier phase scripts are retained where they support validation or dependencies; `reproduce_paper.py` is the paper release gate.
- `tests/`: deterministic unit tests for extraction, probe semantics, graph edits, null-layer preservation, and independent exactness checks.
- `expected/`: frozen derived scientific results. Generated reruns go to the ignored `results/` directory.

The analysis is model-level and structural. It does not test LLM reasoning, column lineage, metric semantics, production runtime, or an operational multiplicity cutoff. The unique-path equality concerns lost reachability L, not endpoint-corrected severity S. The eligible subset is not representative of every Spider project. The main project-level analysis has n=10, including two near-duplicate Workday projects; disclosed sensitivity checks do not remove that limitation. Verification-priority error probabilities are synthetic scenarios, not measured production error rates.

Code is under the Apache-2.0 license in `LICENSE`. Upstream data remain under their own terms; this repository fetches the original inputs rather than bundling them. See `THIRD_PARTY.md`.
