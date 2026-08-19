from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _verify_checksum_file(root: Path, checksum_file: Path) -> None:
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == expected


def test_frozen_submission_and_evidence_checksums_match_repository_bytes():
    artifact = ROOT / "artifacts/submission/guidance_selector_hybrid_v1"
    evidence = ROOT / "automation/evidence/guidance_selector_v1"
    _verify_checksum_file(artifact, artifact / "sha256sums.txt")
    _verify_checksum_file(evidence, evidence / "checksums.sha256")


def test_frozen_hashed_text_uses_canonical_lf():
    paths = (
        ROOT / "configs/submission/guidance_selector_hybrid_v1.json",
        ROOT / "automation/evidence/guidance_selector_v1/episode_records.csv",
        ROOT / "automation/evidence/guidance_selector_v1/paired_200s_results.csv",
        ROOT / "automation/evidence/guidance_selector_v1/checksums.sha256",
    )
    for path in paths:
        assert b"\r\n" not in path.read_bytes(), path
