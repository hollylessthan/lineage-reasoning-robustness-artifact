from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile


RELEASE_URL = (
    "https://github.com/AJamal27891/dw-bench/releases/download/"
    "v1.0.0/dw-bench-zenodo-v1.0.0.zip"
)
RELEASE_SHA256 = "94a4057581467745abba3953add69bd6cc4e9e984b05e7894e9f2238ff9674f9"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_one(root: Path, suffixes: list[str]) -> Path:
    matches = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        norm = p.as_posix()
        if any(norm.endswith(suffix) for suffix in suffixes):
            matches.append(p)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one match for {suffixes}, found {len(matches)}: "
            + ", ".join(str(p) for p in matches[:20])
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "dw-bench-zenodo-v1.0.0.zip"

    with urllib.request.urlopen(RELEASE_URL, timeout=120) as response, archive.open("wb") as out:
        shutil.copyfileobj(response, out)

    digest = _sha256(archive)
    if digest != RELEASE_SHA256:
        raise RuntimeError(
            f"Release archive SHA-256 mismatch: expected {RELEASE_SHA256}, got {digest}"
        )

    extract_dir = args.output_dir / "release"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_dir)

    resolved = {
        "tpc_di_graph": _find_one(
            extract_dir,
            [
                "/datasets/tpc-di/schema_graph.pt",
                "/schemas/tpc-di/schema_graph.pt",
            ],
        ),
        "adventureworks_graph": _find_one(
            extract_dir,
            [
                "/datasets/adventureworks/schema_graph.pt",
                "/schemas/adventureworks/schema_graph.pt",
            ],
        ),
    }

    # Prefer the combined HF-style Tier-1 file if present. Otherwise use the
    # dataset-local QA files included in the release.
    combined = [
        p for p in extract_dir.rglob("test.jsonl")
        if p.as_posix().endswith("/tier1/test.jsonl")
    ]
    if len(combined) == 1:
        resolved["tier1"] = combined[0]
        qa_mode = "combined_jsonl"
    else:
        resolved["tpc_di_qa"] = _find_one(
            extract_dir, ["/datasets/tpc-di/qa_pairs.json"]
        )
        resolved["adventureworks_qa"] = _find_one(
            extract_dir, ["/datasets/adventureworks/qa_pairs.json"]
        )
        qa_mode = "dataset_json"

    payload = {
        "source": "github_release",
        "release_tag": "v1.0.0",
        "release_url": RELEASE_URL,
        "release_sha256": digest,
        "qa_mode": qa_mode,
        "files": {k: str(v.resolve()) for k, v in resolved.items()},
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

