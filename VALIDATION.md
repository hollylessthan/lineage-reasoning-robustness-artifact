# Reproduction validation

The initial export was validated locally on Python 3.12.14 with NetworkX 3.6.1, PyTorch 2.14.0+cpu, and torch-geometric 2.8.0.post1. All 43 unit tests passed. The full reproduction command passed comparisons of all frozen scientific outputs and regenerated manuscript sensitivity checks. See local-reproduction.json for the machine-readable report. CI separately targets the original Python 3.11.16 environment.

The compared frozen outputs are the complete structures, including per-edit and per-edge evidence, not just headline aggregates. No corpus questions, graph-edit rules, null sampler, or scientific implementation were changed during export. The environment dependency versions were pinned. A new orchestration script performs revision and checksum checks and regenerates the previously separately reported sensitivities. The export contains no development history, internal reviews/planning, raw upstream datasets, model API clients, or credentials.

This snapshot is pending author review and CI; it is not yet the artifact-v1.0 release.
