#!/usr/bin/env python3
"""Generate (or verify) the processed-dataset README, figures, and EDA summary.

Usage:
    python generate_preprocessed_readme.py \\
        --processed-root /root/EMS-Project/processed_dataset \\
        --output-dir /root/EMS-Project/readme/preprocessed

    python generate_preprocessed_readme.py \\
        --processed-root /root/EMS-Project/processed_dataset \\
        --output-dir /root/EMS-Project/readme/preprocessed \\
        --verify-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from preprocessing.eda import compute_eda_summary  # noqa: E402
from preprocessing.readme_renderer import render_readme, write_artifacts  # noqa: E402
from preprocessing.storage import sha256_of_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate/verify the processed-dataset README and EDA"
    )
    parser.add_argument(
        "--processed-root",
        default=str(REPO_ROOT / "processed_dataset"),
        help="canonical processed dataset root",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "readme" / "preprocessed"),
        help="directory for README.md, figures/, eda_summary.json",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify stored summary hashes and README text against the current dataset",
    )
    return parser


def verify(processed_root: Path, output_dir: Path) -> int:
    readme_path = output_dir / "README.md"
    summary_path = output_dir / "eda_summary.json"
    if not readme_path.is_file() or not summary_path.is_file():
        print(f"verify failed: missing {readme_path} or {summary_path}", file=sys.stderr)
        return 1
    stored = json.loads(summary_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    # 1. Input-artifact identity must match the stored summary.
    recorded = stored["processed_dataset"]
    for fname, key in [
        ("dataset_metadata.json", "dataset_metadata_sha256"),
        ("image_manifest.csv", "image_manifest_sha256"),
        ("subject_manifest.csv", "subject_manifest_sha256"),
        ("trial_manifest.parquet", "trial_manifest_sha256"),
        ("qc_summary.json", "qc_summary_sha256"),
        ("preprocessing_config.json", "preprocessing_config_sha256"),
    ]:
        path = processed_root / fname
        if not path.is_file():
            errors.append(f"{fname} missing from processed root")
            continue
        actual = sha256_of_file(path)
        if actual != recorded.get(key):
            errors.append(f"{fname} checksum mismatch (stored {recorded.get(key)}, actual {actual})")

    # 2. Stored README must equal the freshly rendered text.
    try:
        rendered = render_readme(stored)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"renderer failed on stored summary: {exc}")
        rendered = None
    if rendered is not None:
        current = readme_path.read_text(encoding="utf-8")
        if current != rendered:
            errors.append("README.md differs from freshly rendered text")

    # 3. Figure files referenced by the stored summary must exist.
    for rel in ["figures/fig_subject_trial_counts.png", "figures/fig_trials_per_subject.png",
                "figures/fig_subject_fixations_by_group.png", "figures/fig_qc_events.png",
                "figures/fig_example_trials.png", "figures/fig_group_average_heatmaps.png"]:
        if not (output_dir / rel).is_file():
            errors.append(f"figure missing: {rel}")

    if errors:
        print("verify FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("verify OK: summary hashes and README match the current processed dataset")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    processed_root = Path(args.processed_root)
    output_dir = Path(args.output_dir)
    if args.verify_only:
        return verify(processed_root, output_dir)

    command = " ".join(sys.argv)
    summary = compute_eda_summary(processed_root, command)
    readme_path, summary_path, figures = write_artifacts(summary, output_dir)
    print(f"wrote {readme_path}")
    print(f"wrote {summary_path}")
    for f in figures:
        print(f"wrote {output_dir / f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
