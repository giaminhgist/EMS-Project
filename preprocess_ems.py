#!/usr/bin/env python3
"""EMS preprocessing CLI.

Usage:
    python preprocess_ems.py --config configs/preprocessing.yaml --dry-run
    python preprocess_ems.py --config configs/preprocessing.yaml
    python preprocess_ems.py --config configs/preprocessing.yaml --resume
    python preprocess_ems.py --config configs/preprocessing.yaml --subjects 000 001 --output-root /path/to/smoke_output
    python preprocess_ems.py --config configs/preprocessing.yaml --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from preprocessing.config import ConfigError, PreprocessingConfig  # noqa: E402
from preprocessing.pipeline import PipelineOptions, run_pipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EMS fixation -> heatmap preprocessing")
    parser.add_argument("--config", required=True, help="path to preprocessing YAML config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="inventory and validate without writing processed arrays",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip subjects whose published arrays match the current config",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace incompatible existing subject outputs after validation",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        metavar="ID",
        help="process only these subject IDs (smoke tests; requires --output-root)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="override the config output_root (required with --subjects)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = PreprocessingConfig.from_yaml(REPO_ROOT / args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.subjects is not None and not args.output_root:
        print(
            "error: --subjects is for smoke tests and must write to a "
            "user-specified --output-root (a partial run must not look like the "
            "canonical dataset)",
            file=sys.stderr,
        )
        return 2

    options = PipelineOptions(
        dry_run=args.dry_run,
        force=args.force,
        resume=args.resume,
        subjects=args.subjects,
        output_root_override=Path(args.output_root) if args.output_root else None,
    )
    try:
        report = run_pipeline(config, options)
    except (ConfigError, ValueError, OSError, ImportError) as exc:
        print(f"preprocessing failed: {exc}", file=sys.stderr)
        return 1

    print(f"config_hash: {report.config_hash}")
    print(f"subjects: {report.n_subjects}")
    print(f"trials observed: {report.n_trials_observed}")
    print(f"trials ok: {report.n_trials_ok}")
    print(f"trials excluded: {report.n_trials_excluded}")
    for note in report.notes:
        print(f"note: {note}")
    if report.dry_run:
        print("dry-run complete: nothing was written")
    else:
        print("preprocessing complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
