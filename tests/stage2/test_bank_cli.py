"""CLI and registry tests for build_stage2_normative_banks.py (guide 03 §5, §12).

The end-to-end build test monkeypatches the checkpoint-resolution step with a
synthetic build so no real EMS data or Stage-1 checkpoint is required.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "build_stage2_normative_banks.py"


def load_cli_module():
    spec = importlib.util.spec_from_file_location("stage2_bank_cli", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage2_bank_cli"] = module
    spec.loader.exec_module(module)
    return module


def write_registry(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "dummy_checkpoint.pt"
    checkpoint.write_bytes(b"dummy checkpoint bytes")
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    registry = {
        "schema_version": 1,
        "evaluation_regime": "pilot_existing_stage1",
        "folds": {
            str(f): {"checkpoint": str(checkpoint), "sha256": sha} for f in range(5)
        },
    }
    path = tmp_path / "stage1_checkpoints.yaml"
    path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    return path


def write_bank_config(tmp_path: Path, **overrides: object) -> Path:
    raw = {
        "seed": 2026,
        "estimator": "mean",
        "epsilon": 1.0e-6,
        "min_samples": 2,
        "batch_size": 64,
        "include_fused_token_bank": False,
        "include_heatmap_token_bank": False,
        "crossfit_splits": 2,
        "crossfit_enabled": True,
        "device": "cpu",
        "output_root": str(tmp_path / "bank"),
        "processed_root": str(REPO_ROOT / "processed_dataset"),
        "dino_root": str(REPO_ROOT / "stimulus_features" / "dino_vits16"),
        "cv_root": str(REPO_ROOT / "CV" / "5fold_seed2026"),
    }
    raw.update(overrides)
    path = tmp_path / "bank.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


class TestRegistry:
    def test_valid_registry_loads(self, tmp_path: Path):
        from stage2.bank_builder import load_checkpoint_registry

        registry = load_checkpoint_registry(write_registry(tmp_path))
        assert registry["evaluation_regime"] == "pilot_existing_stage1"
        assert set(registry["folds"]) == {"0", "1", "2", "3", "4"}

    def test_missing_fold_rejected(self, tmp_path: Path):
        from stage2.bank_builder import BankBuildError, load_checkpoint_registry

        path = write_registry(tmp_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        del raw["folds"]["4"]
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(BankBuildError, match="folds 0-4"):
            load_checkpoint_registry(path)

    def test_bad_regime_rejected(self, tmp_path: Path):
        from stage2.bank_builder import BankBuildError, load_checkpoint_registry

        path = write_registry(tmp_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["evaluation_regime"] = "silently_best_estimate"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(BankBuildError, match="evaluation_regime"):
            load_checkpoint_registry(path)


class TestParser:
    def test_defaults(self):
        module = load_cli_module()
        parser = module.build_parser()
        args = parser.parse_args(["--fold", "0"])
        assert args.fold == "0"
        assert args.verify_only is False
        assert args.include_fused_token_bank is None

    def test_fold_validation(self):
        module = load_cli_module()
        from stage2.bank_builder import BankBuildError

        with pytest.raises(BankBuildError, match="0-4"):
            module._resolve_folds(module.build_parser().parse_args(["--fold", "7"]))


class TestCliEndToEnd:
    def _synthetic_build(self, fold: int):
        from stage2.bank_builder import BankArrays, FoldBankBuild

        def make_arrays() -> BankArrays:
            rng = np.random.default_rng(fold)
            return BankArrays(
                mu_trial=rng.normal(size=(100, 128)).astype(np.float32),
                sigma_trial=(1.0 + rng.random((100, 128))).astype(np.float32),
                count_trial=np.full(100, 5, dtype=np.int32),
            )

        hc = [f"{i:03d}" for i in range(8)]  # 000-007 are real HC labels
        sz = [f"{i:03d}" for i in range(100, 104)]
        rows = []
        for i, sid in enumerate(hc):
            rows.append(
                {
                    "subject_id": sid,
                    "label": 0,
                    "bank_split_id": i % 2,
                    "is_hc_bank_contributor": True,
                    "panel_trial_count": 4,
                }
            )
        for i, sid in enumerate(sz):
            rows.append(
                {
                    "subject_id": sid,
                    "label": 1,
                    "bank_split_id": i % 2,
                    "is_hc_bank_contributor": False,
                    "panel_trial_count": 3,
                }
            )
        assignments = pd.DataFrame(rows)
        excluded = {
            0: {hc[i] for i in range(8) if i % 2 == 0},
            1: {hc[i] for i in range(8) if i % 2 == 1},
        }
        contributors = {
            0: {s for s in hc if s not in excluded[0]},
            1: {s for s in hc if s not in excluded[1]},
        }
        return FoldBankBuild(
            fold=fold,
            full=make_arrays(),
            crossfit={0: make_arrays(), 1: make_arrays()},
            contributors_full=set(hc),
            contributors_crossfit=contributors,
            excluded_crossfit=excluded,
            assignments=assignments,
            n_trials_full=32,
            n_trials_crossfit={0: 16, 1: 16},
        )

    def _provenance(self) -> dict:
        feature_manifest = pd.DataFrame(
            {
                "stimulus_index": list(range(100)),
                "stimulus_id": [f"s{i:03d}.jpg" for i in range(100)],
                "category_id": [i % 4 for i in range(100)],
            }
        )
        return {
            "stage1_checkpoint_path": "/tmp/dummy.pt",
            "stage1_checkpoint_sha256": "a" * 64,
            "stage1_run_id": "run",
            "stage1_config_hash": "b" * 64,
            "processed_dataset_metadata_sha256": "c" * 64,
            "dino_feature_manifest_sha256": "d" * 64,
            "cv_partition_sha256": {"train_trials": "e" * 64, "val_trials": "f" * 64},
            "source_checksums": {},
            "feature_manifest": feature_manifest,
            "forbidden_validation_subject_ids": [f"{i:03d}" for i in range(20, 28)],
        }

    def test_build_skip_and_verify(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        import stage2.bank_builder as bb

        def fake_build(*, fold, registry_entry, evaluation_regime, cfg):
            provenance = self._provenance()
            provenance["stage1_checkpoint_path"] = registry_entry["checkpoint"]
            provenance["stage1_checkpoint_sha256"] = registry_entry["sha256"]
            return self._synthetic_build(fold), provenance

        monkeypatch.setattr(bb, "build_fold_from_checkpoint", fake_build)
        module = load_cli_module()
        registry = write_registry(tmp_path)
        bank_cfg = write_bank_config(tmp_path)
        argv = ["--checkpoint-registry", str(registry), "--config", str(bank_cfg)]

        rc = module.main(argv + ["--fold", "0"])
        assert rc == 0
        fold_dir = tmp_path / "bank" / "fold_0"
        for name in ("mu_trial.npy", "sigma_trial.npy", "count_trial.npy", "feature_manifest.csv", "metadata.json", "audit.json"):
            assert (fold_dir / name).is_file(), name
        assert (fold_dir / "crossfit" / "subject_assignment.csv").is_file()
        for split in (0, 1):
            assert (fold_dir / "crossfit" / f"split_{split}" / "audit.json").is_file()
        manifest = yaml.safe_load((tmp_path / "bank" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["folds"]["0"]["status"] == "complete"
        assert manifest["folds"]["4"]["status"] == "missing"

        # Idempotent: a second run skips the complete fold and succeeds.
        rc = module.main(argv + ["--fold", "0"])
        assert rc == 0

        # Verify-only passes on the intact bank.
        rc = module.main(argv + ["--fold", "0", "--verify-only"])
        assert rc == 0

        # Tampering is detected.
        (fold_dir / "mu_trial.npy").unlink()
        rc = module.main(argv + ["--fold", "0", "--verify-only"])
        assert rc == 1

    def test_verify_only_missing_fold_fails(self, tmp_path: Path):
        module = load_cli_module()
        registry = write_registry(tmp_path)
        bank_cfg = write_bank_config(tmp_path)
        rc = module.main(
            ["--checkpoint-registry", str(registry), "--config", str(bank_cfg), "--fold", "0", "--verify-only"]
        )
        assert rc == 1
