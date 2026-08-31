#!/usr/bin/env python3
"""Frozen DINO stimulus feature extraction CLI.

Usage:
    python extract_dino_features.py \\
        --config configs/dino_vits16.yaml \\
        --image-manifest /root/EMS-Project/processed_dataset/image_manifest.csv \\
        --output-root /root/EMS-Project/stimulus_features

    python extract_dino_features.py --config configs/dino_vits16.yaml ... --resume
    python extract_dino_features.py --config configs/dino_vits16.yaml ... --force
    python extract_dino_features.py --config configs/dino_vits16.yaml ... --verify-only
    python extract_dino_features.py --config configs/dino_vits16.yaml \\
        --stimulus-limit 3 --output-root /tmp/dino_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stimulus_features.config import ConfigError, DINOExtractionConfig  # noqa: E402
from stimulus_features.pipeline import ExtractionOptions, run_extraction  # noqa: E402
from stimulus_features.storage import FeaturePaths, StorageError  # noqa: E402
from stimulus_features.validate import verify_only  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen DINO stimulus feature extraction")
    parser.add_argument("--config", required=True, help="path to DINO extraction YAML config")
    parser.add_argument(
        "--image-manifest",
        default=None,
        help="override the image manifest path (default: from config)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="override the config output_root (required with --stimulus-limit)",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="verify and continue existing output")
    parser.add_argument("--force", action="store_true", help="replace existing output after validation")
    parser.add_argument("--verify-only", action="store_true", help="verify existing artifacts only")
    parser.add_argument(
        "--stimulus-limit",
        type=int,
        default=None,
        help="extract only the first N stimuli (smoke tests; requires --output-root)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = DINOExtractionConfig.from_yaml(REPO_ROOT / args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    # CLI overrides are recorded by replacing the config object fields.
    if args.image_manifest:
        cfg = DINOExtractionConfig(**{**cfg.to_dict(), "image_manifest": str(REPO_ROOT / args.image_manifest)})

    out_root_override = Path(args.output_root) if args.output_root else None
    out_dir = (out_root_override / cfg.output_subdir) if out_root_override else cfg.feature_output_dir
    paths = FeaturePaths.for_dir(out_dir)

    if args.verify_only:
        try:
            result = verify_only(cfg, paths, cfg.image_manifest)
        except (StorageError, ValueError, OSError) as exc:
            print(f"verify FAILED: {exc}", file=sys.stderr)
            return 1
        print("verify OK:", result)
        return 0

    if args.stimulus_limit is not None and not args.output_root:
        print(
            "error: --stimulus-limit is for smoke tests and requires --output-root",
            file=sys.stderr,
        )
        return 2

    options = ExtractionOptions(
        device=args.device,
        batch_size=args.batch_size,
        resume=args.resume,
        force=args.force,
        stimulus_limit=args.stimulus_limit,
        output_root_override=out_root_override,
    )
    try:
        report = run_extraction(cfg, options)
    except (ConfigError, StorageError, ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"extraction failed: {exc}", file=sys.stderr)
        return 1
    print(f"status: {report['status']}")
    print(f"stimuli: {report['n_stimuli']}")
    print(f"output: {report['output_dir']}")
    print(f"elapsed: {report['elapsed_seconds']:.1f}s")
    for note in report.get("notes", []):
        print(f"note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
