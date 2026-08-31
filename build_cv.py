#!/usr/bin/env python3
"""Build (or verify) the five-fold subject-level cross-validation split.

Usage:
    python build_cv.py \\
        --config configs/cv_5fold.yaml \\
        --subject-manifest /root/EMS-Project/processed_dataset/subject_manifest.csv \\
        --trial-manifest /root/EMS-Project/processed_dataset/trial_manifest.parquet \\
        --output-root /root/EMS-Project/CV

    python build_cv.py --config configs/cv_5fold.yaml ... --verify-only
    python build_cv.py --config configs/cv_5fold.yaml ... --force
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cv.build_subject_folds import (  # noqa: E402
    CVError,
    Population,
    build_assignments,
    fold_subject_partitions,
    fold_summaries,
    fold_trial_partitions,
)
from cv.config import CVConfig, ConfigError  # noqa: E402
from cv.validate_folds import run_all_checks  # noqa: E402
from preprocessing.storage import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    sha256_of_file,
)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    import io

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    atomic_write_bytes(path, buf.getvalue().encode("utf-8"))


def build_split(
    cfg: CVConfig, force: bool = False
) -> dict[str, Any]:
    """Build the versioned CV directory atomically; return a report dict."""
    out_dir = cfg.output_dir
    existing_config = None
    if out_dir.exists():
        cfg_path = out_dir / "cv_config.json"
        if cfg_path.is_file():
            existing_config = json.loads(cfg_path.read_text(encoding="utf-8"))
        if existing_config is not None and existing_config.get("config_hash") != cfg.config_hash():
            raise CVError(
                f"existing split {out_dir} was built with config hash "
                f"{existing_config.get('config_hash')}; refusing to regenerate under "
                f"the same directory. Use a new versioned directory or approve "
                f"replacement of an identical configuration."
            )
        if existing_config is None and not force:
            raise CVError(f"{out_dir} exists without cv_config.json; use --force to replace")

    population = Population.load(cfg)
    assignments = build_assignments(cfg, population.subjects)
    subject_partitions = fold_subject_partitions(cfg, assignments)
    trial_partitions = fold_trial_partitions(assignments, population.trials, cfg.n_splits)
    summaries = fold_summaries(cfg, assignments, trial_partitions)
    checks = run_all_checks(cfg, assignments, population.trials)
    if not checks["all_checks_pass"]:
        raise CVError(f"CV validation checks failed: {checks['failures']}")

    # Stage the whole directory, then publish atomically.
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    staging = out_dir.parent / f".{out_dir.name}.staging_{os.getpid()}_{int(time.time() * 1000)}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for fold in range(cfg.n_splits):
            fold_dir = staging / f"fold_{fold}"
            fold_dir.mkdir()
            train_subjects, val_subjects = subject_partitions[fold]
            train_trials, val_trials = trial_partitions[fold]
            _write_csv(fold_dir / "train_subjects.csv", train_subjects)
            _write_csv(fold_dir / "val_subjects.csv", val_subjects)
            train_trials.to_parquet(fold_dir / "train_trials.parquet", index=False)
            val_trials.to_parquet(fold_dir / "val_trials.parquet", index=False)

        _write_csv(staging / "fold_assignments.csv", assignments)
        atomic_write_json(staging / "cv_config.json", {**cfg.to_dict(), "config_hash": cfg.config_hash()})
        atomic_write_json(staging / "validation_report.json", checks)

        import numpy as np

        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_hash": cfg.config_hash(),
            "seed": cfg.random_state,
            "git_commit": _git_commit(),
            "library_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": _sklearn_version(),
            },
            "input_checksums": {
                "subject_manifest_sha256": sha256_of_file(cfg.subject_manifest),
                "trial_manifest_sha256": sha256_of_file(cfg.trial_manifest),
                "source_inventory_sha256": sha256_of_file(cfg.source_inventory),
            },
            "fold_assignments_sha256": sha256_of_file(staging / "fold_assignments.csv"),
            "population": {
                "n_subjects": int(len(assignments)),
                "n_hc": int((assignments.label == 0).sum()),
                "n_sz": int((assignments.label == 1).sum()),
                "n_trials": int(len(population.trials)),
            },
            "folds": summaries,
        }
        atomic_write_json(staging / "cv_metadata.json", metadata)

        old = None
        if out_dir.exists():
            old = out_dir.parent / f".{out_dir.name}.old_{os.getpid()}_{int(time.time() * 1000)}"
            os.replace(out_dir, old)
        try:
            os.replace(staging, out_dir)
        except BaseException:
            if old is not None and old.exists():
                os.replace(old, out_dir)
            raise
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {"output_dir": str(out_dir), "population": metadata["population"], "folds": summaries}


def verify_only(cfg: CVConfig) -> dict[str, Any]:
    """Recompute all checks and checksums against the stored split."""
    out_dir = cfg.output_dir
    for required in ["fold_assignments.csv", "cv_config.json", "cv_metadata.json", "validation_report.json"]:
        if not (out_dir / required).is_file():
            raise CVError(f"missing {required} in {out_dir}")
    stored_cfg = json.loads((out_dir / "cv_config.json").read_text(encoding="utf-8"))
    if stored_cfg.get("config_hash") != cfg.config_hash():
        raise CVError(
            f"stored config hash {stored_cfg.get('config_hash')} != current {cfg.config_hash()}"
        )
    stored_meta = json.loads((out_dir / "cv_metadata.json").read_text(encoding="utf-8"))
    for key, path in [
        ("subject_manifest_sha256", cfg.subject_manifest),
        ("trial_manifest_sha256", cfg.trial_manifest),
        ("source_inventory_sha256", cfg.source_inventory),
    ]:
        actual = sha256_of_file(path)
        if stored_meta["input_checksums"].get(key) != actual:
            raise CVError(f"{key} mismatch: stored {stored_meta['input_checksums'].get(key)}, actual {actual}")

    assignments = pd.read_csv(
        out_dir / "fold_assignments.csv",
        dtype={"subject_id": str, "group": str},
    )
    if stored_meta["fold_assignments_sha256"] != sha256_of_file(out_dir / "fold_assignments.csv"):
        raise CVError("fold_assignments.csv checksum mismatch")

    # Recompute the assignment from the current inputs and require identity.
    population = Population.load(cfg)
    expected = build_assignments(cfg, population.subjects)
    if not expected.equals(assignments):
        raise CVError("stored fold assignments differ from a fresh deterministic build")

    checks = run_all_checks(cfg, assignments, population.trials)
    if not checks["all_checks_pass"]:
        raise CVError(f"stored split fails checks: {checks['failures']}")

    # Partition files resolve every subject/trial through explicit IDs.
    for fold in range(cfg.n_splits):
        fold_dir = out_dir / f"fold_{fold}"
        for name in ["train_subjects.csv", "val_subjects.csv", "train_trials.parquet", "val_trials.parquet"]:
            if not (fold_dir / name).is_file():
                raise CVError(f"missing {fold_dir / name}")
        train_subjects = pd.read_csv(fold_dir / "train_subjects.csv", dtype={"subject_id": str})
        val_subjects = pd.read_csv(fold_dir / "val_subjects.csv", dtype={"subject_id": str})
        train_trials = pd.read_parquet(fold_dir / "train_trials.parquet")
        val_trials = pd.read_parquet(fold_dir / "val_trials.parquet")
        if set(train_subjects.subject_id) & set(val_subjects.subject_id):
            raise CVError(f"fold {fold}: subject overlap in partition files")
        if set(train_trials.subject_id) - set(train_subjects.subject_id):
            raise CVError(f"fold {fold}: train trials reference subjects outside train_subjects.csv")
        if set(val_trials.subject_id) - set(val_subjects.subject_id):
            raise CVError(f"fold {fold}: val trials reference subjects outside val_subjects.csv")
        # Stage-1 HC views are defined as the HC rows of these same partitions;
        # verify the stored HC counts match the files.
        stored_fold = next(f for f in stored_meta["folds"] if f["fold"] == fold)
        if stored_fold["stage1_train_hc_subjects"] != int((train_subjects.group == "HC").sum()):
            raise CVError(f"fold {fold}: stored stage1 train HC subject count mismatch")
        if stored_fold["stage1_val_hc_subjects"] != int((val_subjects.group == "HC").sum()):
            raise CVError(f"fold {fold}: stored stage1 val HC subject count mismatch")
    return {
        "status": "ok",
        "config_hash": cfg.config_hash(),
        "n_subjects": int(len(assignments)),
        "checks": checks["all_checks_pass"],
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(REPO_ROOT), check=False, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _sklearn_version() -> str:
    import sklearn

    return sklearn.__version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Five-fold subject-level CV builder")
    parser.add_argument("--config", required=True, help="path to CV YAML config")
    parser.add_argument("--subject-manifest", default=None)
    parser.add_argument("--trial-manifest", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = CVConfig.from_yaml(REPO_ROOT / args.config)
        overrides = {}
        if args.subject_manifest:
            overrides["subject_manifest"] = str(REPO_ROOT / args.subject_manifest)
        if args.trial_manifest:
            overrides["trial_manifest"] = str(REPO_ROOT / args.trial_manifest)
        if args.output_root:
            overrides["output_root"] = str(REPO_ROOT / args.output_root)
        if overrides:
            cfg = CVConfig(**{**cfg.to_dict(), **overrides})
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    try:
        if args.verify_only:
            result = verify_only(cfg)
            print("verify OK:", result)
            return 0
        report = build_split(cfg, force=args.force)
    except (CVError, ConfigError, ValueError, OSError) as exc:
        print(f"CV build failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {report['output_dir']}")
    print(f"population: {report['population']}")
    for f in report["folds"]:
        print(
            f"  fold {f['fold']}: train {f['n_train_subjects']} subjects "
            f"({f['train_hc']} HC / {f['train_sz']} SZ), "
            f"val {f['n_val_subjects']} subjects ({f['val_hc']} HC / {f['val_sz']} SZ); "
            f"stage1 HC: train {f['stage1_train_hc_subjects']} subj / "
            f"{f['stage1_train_hc_trials']} trials, "
            f"val {f['stage1_val_hc_subjects']} subj / {f['stage1_val_hc_trials']} trials"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
