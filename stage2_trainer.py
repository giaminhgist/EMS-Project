"""Stage-2 root training entry point (guide 07 §3-§5).

Thin CLI: argument parsing, configuration resolution, fold iteration and exit
codes. All trainable behavior lives in ``src/stage2/``. Exit codes: 0 = ok,
1 = usage/config/verification error, 2 = training fold failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from stage2.ablations import (  # noqa: E402
    ABLATIONS,
    BASE_CONFIG_DEFAULT,
    ResolvedConfig,
    resolve_ablation_config,
    validate_bank_capabilities,
)
from stage2.config import (  # noqa: E402
    ConfigError,
    Stage2Config,
    config_hash,
    load_yaml_dict,
)
from stage2.trainer import (  # noqa: E402
    Stage2Trainer,
    TrainerError,
    TrainerLimits,
    make_run_id,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage-2 HC-normative subject classification trainer"
    )
    parser.add_argument("--config", default=str(BASE_CONFIG_DEFAULT))
    parser.add_argument("--fold", default="0", help="{0,1,2,3,4,all}")
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--evaluation-regime", default="pilot_existing_stage1",
        choices=["pilot_existing_stage1", "strict_nested_stage1"],
    )
    parser.add_argument("--stage1-checkpoint", default=None)
    parser.add_argument("--bank-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--load-stage2-weights", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="mark the run as a smoke run")
    parser.add_argument("--max-train-subjects", type=int, default=None)
    parser.add_argument("--max-val-subjects", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="dotted-key override applied after the ablation overlay (repeatable)",
    )
    return parser


def parse_overrides(items: list[str]) -> dict[str, Any]:
    from stage2.config import coerce_cli_value

    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"override must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = coerce_cli_value(value.strip())
    return overrides


def resolve_folds(fold_arg: str) -> list[int]:
    if fold_arg == "all":
        return [0, 1, 2, 3, 4]
    try:
        fold = int(fold_arg)
    except ValueError:
        raise ConfigError(f"--fold must be one of {{0,1,2,3,4,all}}, got {fold_arg!r}")
    if not 0 <= fold <= 4:
        raise ConfigError("--fold must be in [0, 4] or 'all'")
    return [fold]


def _fold_of_checkpoint_path(path: Path) -> int:
    name = path.name  # e.g. last_stage2_fold0.pt
    for f in range(5):
        if f"fold{f}" in name:
            return f
    raise ConfigError(f"cannot derive the fold from checkpoint path {path}")


def resolve_config_for_run(args: argparse.Namespace, fold: int) -> ResolvedConfig | Stage2Config:
    """base -> optional overlay -> CLI overrides -> fold/seed/runtime fields."""
    overrides = parse_overrides(args.override)
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.bank_root is not None:
        overrides["bank.root"] = args.bank_root
    if args.output_root is not None:
        overrides["paths.output_root"] = args.output_root
    if args.num_workers is not None:
        overrides["runtime.num_workers"] = args.num_workers
    if args.ablation:
        if args.evaluation_regime != "pilot_existing_stage1":
            raise ConfigError(
                "strict_nested_stage1 is not available: no Stage-1 checkpoint "
                "selected without the outer held-out fold exists under the "
                "approved pilot_existing_stage1 artifacts (Phase-0 decision A1)"
            )
        resolved = resolve_ablation_config(
            args.config, args.ablation, overrides=overrides, fold=fold
        )
        resolved.config.validate()
        return resolved
    raw = load_yaml_dict(args.config)
    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            node: dict[str, Any] = raw
            for part in parts[:-1]:
                if not isinstance(node.get(part), dict):
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
    raw["fold"] = fold
    cfg = Stage2Config.from_dict(raw, source=str(args.config))
    return cfg


def verify_stage1_checkpoint_override(args: argparse.Namespace, fold: int) -> None:
    """A user-supplied Stage-1 checkpoint must be byte-identical to the
    registry entry for this fold (never a silent fallback)."""
    if args.stage1_checkpoint is None:
        return
    from stage2.bank_builder import load_checkpoint_registry
    from stage2.contracts import sha256_of_file

    base_raw = load_yaml_dict(args.config)
    registry_file = base_raw.get("bank", {}).get("checkpoint_registry")
    if not registry_file:
        raise ConfigError("--stage1-checkpoint requires bank.checkpoint_registry in the config")
    registry = load_checkpoint_registry(Path(registry_file))
    entry = registry["folds"][str(fold)]
    given = Path(args.stage1_checkpoint)
    if not given.is_file():
        raise ConfigError(f"--stage1-checkpoint not found: {given}")
    given_sha = sha256_of_file(given)
    if given_sha != str(entry["sha256"]):
        raise ConfigError(
            f"--stage1-checkpoint SHA-256 {given_sha} does not match the approved "
            f"registry entry {entry['sha256']} for fold {fold}; update the registry "
            f"instead of overriding at the CLI"
        )


def run_verify_only(cfg: Stage2Config, fold: int, device: str) -> None:
    from stage2.bank import NormativeBankStore
    from stage2.dataset import Stage2SubjectDataset
    from stage2.model import Stage2Model

    bank_store = NormativeBankStore(cfg, fold, device=device)
    if cfg.ablation != "base" and cfg.ablation in ABLATIONS:
        validate_bank_capabilities(ABLATIONS[cfg.ablation], bank_store)
    train_ds = Stage2SubjectDataset(cfg, fold, "train", bank_store=bank_store)
    val_ds = Stage2SubjectDataset(cfg, fold, "val", bank_store=bank_store)
    model = Stage2Model(cfg, bank_store, device=device)
    report = model.parameter_report()
    print(f"fold {fold}: verification OK")
    print(f"  ablation {cfg.ablation} | bank regime {bank_store.evaluation_regime} | "
          f"train_mode {bank_store.train_mode}")
    print(f"  train subjects {len(train_ds)} | val subjects {len(val_ds)}")
    print(f"  stage1 checkpoint {bank_store.registry_entry['checkpoint']}")
    print(f"  stage1 checkpoint sha256 {bank_store.registry_entry['sha256']}")
    print(f"  parameters: total {report['total']} | trainable {report['trainable']} | "
          f"frozen {report['frozen']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        limits = TrainerLimits(
            max_train_subjects=args.max_train_subjects,
            max_val_subjects=args.max_val_subjects,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
        )
        has_limits = any(v is not None for v in limits.to_dict().values())
        if has_limits and not (args.dry_run or args.smoke):
            raise ConfigError(
                "smoke-limit flags (--max-*) require --dry-run or --smoke"
            )
        if args.resume and args.load_stage2_weights:
            raise ConfigError("--resume and --load-stage2-weights are mutually exclusive")

        if args.resume:
            return run_resume(args, limits)
        if args.load_stage2_weights:
            return run_weight_init(args, limits)

        folds = resolve_folds(args.fold)
        output_root = Path(
            args.output_root
            or load_yaml_dict(args.config).get("paths", {}).get("output_root")
            or REPO_ROOT / "outputs" / "stage2"
        )
        exit_code = 0
        for fold in folds:
            resolved = resolve_config_for_run(args, fold)
            cfg = resolved.config if isinstance(resolved, ResolvedConfig) else resolved
            if isinstance(resolved, ResolvedConfig):
                config_hash_value = resolved.config_hash
                spec = resolved.spec
                ablation_spec = {
                    "name": spec.name,
                    "scientific_question": spec.scientific_question,
                    "declared_changes": list(spec.declared_changes),
                    "required_bank_capabilities": list(spec.required_bank_capabilities),
                    "forbidden_with": list(spec.forbidden_with),
                    "interpretation": spec.interpretation,
                    "is_negative_control": spec.is_negative_control,
                    "reference": spec.reference,
                }
                ablation_diff = resolved.diff_entries
            else:
                config_hash_value = config_hash(cfg.to_dict())
                ablation_spec = None
                ablation_diff = None
            verify_stage1_checkpoint_override(args, fold)

            if args.verify_only:
                run_verify_only(cfg, fold, args.device)
                continue

            run_id = make_run_id(
                cfg.experiment_name,
                cfg.ablation,
                cfg.seed,
                config_hash_value,
                is_smoke=args.smoke or args.dry_run,
            )
            if args.dry_run:
                run_id += "_dryrun"
            run_root = output_root / run_id
            if run_root.exists():
                # A completed run is never overwritten; a partial directory
                # from a failed initialization is removed and retried.
                if (run_root / "run_metadata.json").is_file():
                    raise ConfigError(
                        f"completed run directory already exists (never overwritten): {run_root}"
                    )
                import shutil

                shutil.rmtree(run_root)
            trainer = Stage2Trainer(
                cfg=cfg,
                run_root=run_root,
                run_id=run_id,
                config_hash=config_hash_value,
                ablation_spec=ablation_spec,
                ablation_diff=ablation_diff,
                device=args.device,
                limits=limits,
                is_smoke=args.smoke,
                deterministic=args.deterministic,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                summary = trainer.train()
                print(
                    f"fold {fold}: training complete "
                    f"(best_epoch {summary['best_epoch']}, "
                    f"best_metric {summary['best_metric']}, "
                    f"stop_reason {summary['stop_reason']})"
                )
            else:
                print(f"fold {fold}: dry run complete ({run_root})")
        return exit_code
    except TrainerError as exc:
        print(f"stage2 training failed: {exc}", file=sys.stderr)
        return 2
    except (ConfigError, ValueError, OSError) as exc:
        print(f"stage2 error: {exc}", file=sys.stderr)
        return 1


def run_resume(args: argparse.Namespace, limits: TrainerLimits) -> int:
    """Exact resume: derive run id/fold/config from the checkpoint itself."""
    from stage2.checkpoint import load_stage2_checkpoint

    path = Path(args.resume)
    fold = _fold_of_checkpoint_path(path)
    contents = load_stage2_checkpoint(path, device=args.device)
    meta = contents.meta
    run_id = str(meta["run_id"])
    config_hash_value = str(meta["config_hash"])
    cfg = Stage2Config.from_dict(meta["config_resolved"], source=str(path))
    cfg = Stage2Config.from_dict({**cfg.to_dict(), "fold": fold})
    resolved = resolve_config_for_run(args, fold)
    current_cfg = resolved.config if isinstance(resolved, ResolvedConfig) else resolved
    if config_hash(current_cfg.to_dict()) != config_hash_value:
        raise ConfigError(
            "resume rejected: the current configuration does not match the "
            "checkpoint — use --load-stage2-weights for a different configuration"
        )
    run_root = path.parents[2]
    trainer = Stage2Trainer(
        cfg=cfg,
        run_root=run_root,
        run_id=run_id,
        config_hash=config_hash_value,
        ablation_spec=meta.get("ablation_spec"),
        ablation_diff=meta.get("ablation_diff"),
        device=args.device,
        limits=limits,
        is_smoke=args.smoke,
        deterministic=args.deterministic,
        resume_path=path,
    )
    summary = trainer.train()
    print(
        f"fold {fold}: resumed training complete "
        f"(best_epoch {summary['best_epoch']}, stop_reason {summary['stop_reason']})"
    )
    return 0


def run_weight_init(args: argparse.Namespace, limits: TrainerLimits) -> int:
    """Weight-only initialization into a fresh run."""
    fold = resolve_folds(args.fold)[0]
    resolved = resolve_config_for_run(args, fold)
    cfg = resolved.config if isinstance(resolved, ResolvedConfig) else resolved
    if isinstance(resolved, ResolvedConfig):
        config_hash_value = resolved.config_hash
        spec = resolved.spec
        ablation_spec = {
            "name": spec.name,
            "scientific_question": spec.scientific_question,
            "declared_changes": list(spec.declared_changes),
            "required_bank_capabilities": list(spec.required_bank_capabilities),
            "forbidden_with": list(spec.forbidden_with),
            "interpretation": spec.interpretation,
            "is_negative_control": spec.is_negative_control,
            "reference": spec.reference,
        }
        ablation_diff = resolved.diff_entries
    else:
        config_hash_value = config_hash(cfg.to_dict())
        ablation_spec = None
        ablation_diff = None
    output_root = Path(cfg.paths.output_root or REPO_ROOT / "outputs" / "stage2")
    run_id = make_run_id(
        cfg.experiment_name, cfg.ablation, cfg.seed, config_hash_value, is_smoke=args.smoke
    )
    run_root = output_root / run_id
    if run_root.exists():
        raise ConfigError(f"run directory already exists (never overwritten): {run_root}")
    trainer = Stage2Trainer(
        cfg=cfg,
        run_root=run_root,
        run_id=run_id,
        config_hash=config_hash_value,
        ablation_spec=ablation_spec,
        ablation_diff=ablation_diff,
        device=args.device,
        limits=limits,
        is_smoke=args.smoke,
        deterministic=args.deterministic,
        load_weights_path=Path(args.load_stage2_weights),
    )
    summary = trainer.train()
    print(
        f"fold {fold}: weight-initialized training complete "
        f"(best_epoch {summary['best_epoch']}, stop_reason {summary['stop_reason']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
