#!/usr/bin/env python3
"""Stage-2 normative bank builder CLI (Phase 1, guide 03 §5).

Usage:
    python build_stage2_normative_banks.py \
        --checkpoint-registry configs/stage2/stage1_checkpoints.yaml \
        --config configs/stage2/bank.yaml --fold all

    python build_stage2_normative_banks.py \
        --checkpoint-registry configs/stage2/stage1_checkpoints.yaml \
        --config configs/stage2/bank.yaml --fold 0 --output-root <smoke root> \
        --no-include-fused-token-bank --no-include-heatmap-token-bank

    python build_stage2_normative_banks.py \
        --checkpoint-registry configs/stage2/stage1_checkpoints.yaml \
        --config configs/stage2/bank.yaml --fold all --verify-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stage2.bank_builder import (  # noqa: E402
    BankBuildError,
    BankConfig,
    BankVerifyError,
    build_all_folds,
    verify_all,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and verify five-fold HC normative banks for Stage 2"
    )
    parser.add_argument(
        "--checkpoint-registry",
        default=str(REPO_ROOT / "configs" / "stage2" / "stage1_checkpoints.yaml"),
        help="approved Stage-1 checkpoint registry (default: configs/stage2/stage1_checkpoints.yaml)",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "stage2" / "bank.yaml"),
        help="bank configuration (default: configs/stage2/bank.yaml)",
    )
    parser.add_argument("--fold", default="all", help="fold index 0-4, or 'all'")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="override bank.yaml device")
    parser.add_argument("--output-root", default=None, help="override bank.yaml output_root")
    parser.add_argument(
        "--include-fused-token-bank",
        dest="include_fused_token_bank",
        action="store_true",
        default=None,
        help="enable fused-token bank arrays",
    )
    parser.add_argument(
        "--no-include-fused-token-bank",
        dest="include_fused_token_bank",
        action="store_false",
        help="disable fused-token bank arrays",
    )
    parser.add_argument(
        "--include-heatmap-token-bank",
        dest="include_heatmap_token_bank",
        action="store_true",
        default=None,
        help="enable same-space heatmap-token bank arrays",
    )
    parser.add_argument(
        "--no-include-heatmap-token-bank",
        dest="include_heatmap_token_bank",
        action="store_false",
        help="disable same-space heatmap-token bank arrays",
    )
    parser.add_argument("--crossfit-splits", type=int, default=None, help="override crossfit_splits")
    parser.add_argument("--no-crossfit", action="store_true", help="disable crossfit banks")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="read-only verification of existing banks; never writes",
    )
    parser.add_argument(
        "--overwrite-incomplete",
        action="store_true",
        help="rebuild a fold directory that exists but is incomplete",
    )
    return parser


def _resolve_cfg(args: argparse.Namespace) -> BankConfig:
    cfg = BankConfig.from_yaml(args.config)
    overrides = cfg.to_dict()
    if args.output_root is not None:
        overrides["output_root"] = args.output_root
    if args.device is not None:
        overrides["device"] = args.device
    if args.include_fused_token_bank is not None:
        overrides["include_fused_token_bank"] = args.include_fused_token_bank
    if args.include_heatmap_token_bank is not None:
        overrides["include_heatmap_token_bank"] = args.include_heatmap_token_bank
    if args.crossfit_splits is not None:
        overrides["crossfit_splits"] = args.crossfit_splits
    if args.no_crossfit:
        overrides["crossfit_enabled"] = False
    return BankConfig.from_dict(overrides, source="<cli>")


def _resolve_folds(args: argparse.Namespace) -> list[int]:
    if args.fold == "all":
        return list(range(5))
    try:
        fold = int(args.fold)
    except ValueError:
        raise BankBuildError(f"--fold must be 0-4 or 'all', got {args.fold!r}")
    if not 0 <= fold <= 4:
        raise BankBuildError(f"--fold must be 0-4 or 'all', got {fold}")
    return [fold]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        folds = _resolve_folds(args)
        cfg = _resolve_cfg(args)
        if args.verify_only:
            results = verify_all(
                registry_path=Path(args.checkpoint_registry),
                cfg=cfg,
                folds=folds,
            )
            print(f"verify-only OK for folds {folds}")
            for key in sorted(k for k in results if isinstance(k, str)):
                if key == "root_manifest":
                    statuses = {
                        k: v.get("status") for k, v in results[key].get("folds", {}).items()
                    }
                    print(f"  root manifest fold statuses: {statuses}")
            return 0
        reports = build_all_folds(
            registry_path=Path(args.checkpoint_registry),
            cfg=cfg,
            folds=folds,
            overwrite_incomplete=args.overwrite_incomplete,
        )
        for report in reports:
            print(
                f"fold {report['fold']}: status={report['status']}"
                + (f" n_trials_full={report['n_trials_full']}" if report["status"] == "built" else "")
                + (
                    f" n_trials_crossfit={report['n_trials_crossfit']}"
                    if report["status"] == "built"
                    else ""
                )
            )
        return 0
    except (BankBuildError, BankVerifyError, ValueError, OSError) as exc:
        print(f"bank build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
