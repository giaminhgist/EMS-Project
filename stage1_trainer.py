#!/usr/bin/env python3
"""Stage-1 trainer CLI (contract §9).

Usage:
    python stage1_trainer.py --config configs/stage1/base.yaml --fold 0
    python stage1_trainer.py --config configs/stage1/base.yaml --fold all
    python stage1_trainer.py --config configs/stage1/base.yaml --fold 0 \\
        --max-epochs 2 --max-train-batches 3 --max-val-batches 3 --run-name smoke
    python stage1_trainer.py --config configs/stage1/base.yaml --fold 0 --dry-run
    python stage1_trainer.py --resume outputs/stage1/<run>/fold_0/checkpoints/last_stage1_fold0.pt
    python stage1_trainer.py --config configs/stage1/base.yaml --fold 0 \\
        --load-stage1-weights outputs/stage1/<run>/fold_0/checkpoints/best_stage1_fold0.pt
    python stage1_trainer.py --build-norm-bank \\
        --checkpoint outputs/stage1/<run>/fold_0/checkpoints/best_stage1_fold0.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stage1.config import ConfigError, Stage1Config  # noqa: E402
from stage1.trainer import (  # noqa: E402
    TrainLimits,
    TrainingError,
    build_norm_bank_from_checkpoint,
    run_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-1 HC normative encoder trainer")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "stage1" / "base.yaml"))
    parser.add_argument("--ablation", default=None, help="named ablation overlay (configs/stage1/ablations/<name>.yaml)")
    parser.add_argument("--fold", type=str, default=None, help="fold index 0-4, or 'all'")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--resume", default=None, help="checkpoint path for exact resume")
    parser.add_argument("--load-stage1-weights", default=None, help="checkpoint path for weight-only init")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--run-name", default=None, help="smoke run label")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--build-norm-bank", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="checkpoint for --build-norm-bank")
    parser.add_argument("--norm-bank-out", default=None)
    return parser


def _load_cfg(args: argparse.Namespace) -> Stage1Config:
    base = REPO_ROOT / args.config
    cfg = Stage1Config.load_base_with_ablation(base, args.ablation)
    if args.fold is not None and args.fold != "all":
        cfg = Stage1Config(**{**cfg.to_dict(), "fold": int(args.fold)})
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.build_norm_bank:
            if not args.checkpoint:
                print("error: --build-norm-bank requires --checkpoint", file=sys.stderr)
                return 2
            cfg = _load_cfg(args)
            meta = build_norm_bank_from_checkpoint(
                Path(args.checkpoint),
                cfg=cfg,
                device=args.device,
                output_dir=Path(args.norm_bank_out) if args.norm_bank_out else None,
            )
            print(f"norm bank written: {meta}")
            return 0

        if args.resume and args.fold == "all":
            print("error: --resume requires a single --fold", file=sys.stderr)
            return 2
        cfg = _load_cfg(args)
        folds = list(range(5)) if args.fold == "all" else [cfg.fold]
        for fold in folds:
            fold_cfg = Stage1Config(**{**cfg.to_dict(), "fold": fold})
            outcome = run_training(
                fold_cfg,
                device=args.device,
                limits=TrainLimits(
                    max_epochs=args.max_epochs,
                    max_train_batches=args.max_train_batches,
                    max_val_batches=args.max_val_batches,
                    run_name=args.run_name,
                ),
                resume_path=Path(args.resume) if args.resume else None,
                weights_path=Path(args.load_stage1_weights) if args.load_stage1_weights else None,
                dry_run=args.dry_run,
            )
            print(f"fold {fold}: run_id={outcome.run_id} epochs={outcome.epochs_completed} "
                  f"best_epoch={outcome.best_epoch} best_val_loss={outcome.best_val_loss}")
            for note in outcome.notes:
                print(f"  note: {note}")
        return 0
    except (ConfigError, TrainingError, ValueError, OSError) as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
