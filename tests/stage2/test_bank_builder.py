"""Unit tests for the Stage-2 bank builder primitives (guide 03 §13).

Synthetic tensors only; no real EMS data, checkpoints or models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stage2.bank_builder import (
    BankBuildError,
    BankConfig,
    BankVerifyError,
    StreamingAccumulator,
    build_feature_manifest,
    build_metadata,
    publish_bank_dir,
)
from stage2.contracts import expected_array_shapes
from stage2.bank_builder import BankArrays


def make_accumulator(n_stimuli: int = 5, feature_shape: tuple[int, ...] = (3,)) -> StreamingAccumulator:
    return StreamingAccumulator(n_stimuli, feature_shape)


class TestStreamingAccumulator:
    def test_float64_mean_matches_numpy(self):
        rng = np.random.default_rng(0)
        acc = make_accumulator(n_stimuli=4, feature_shape=(2,))
        feat = rng.normal(size=(12, 2))
        si = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 0, 2])
        acc.add(si, feat)
        mu, sigma, count = acc.finalize(epsilon=1e-6)
        assert mu.dtype == np.float32 and sigma.dtype == np.float32
        assert count.dtype == np.int32
        assert list(count) == [3, 3, 2, 4]
        for s in range(4):
            group = feat[si == s]
            np.testing.assert_allclose(mu[s], group.mean(axis=0), rtol=1e-6, atol=1e-6)
            expected_var = np.maximum((group**2).mean(axis=0) - group.mean(axis=0) ** 2, 1e-12)
            np.testing.assert_allclose(sigma[s], np.sqrt(expected_var), rtol=1e-5, atol=1e-5)

    def test_duplicate_stimulus_indices_accumulate(self):
        acc = make_accumulator(n_stimuli=2, feature_shape=(1,))
        acc.add(np.array([0, 0, 1]), np.array([[1.0], [3.0], [5.0]]))
        mu, _, count = acc.finalize(epsilon=1e-6)
        assert count[0] == 2 and count[1] == 1
        assert mu[0, 0] == 2.0 and mu[1, 0] == 5.0

    def test_out_of_range_stimulus_index_raises(self):
        acc = make_accumulator(n_stimuli=3, feature_shape=(1,))
        with pytest.raises(BankBuildError, match="out of manifest range"):
            acc.add(np.array([0, 3]), np.ones((2, 1)))
        with pytest.raises(BankBuildError, match="out of manifest range"):
            acc.add(np.array([-1]), np.ones((1, 1)))

    def test_shape_mismatch_raises(self):
        acc = make_accumulator(n_stimuli=3, feature_shape=(2,))
        with pytest.raises(BankBuildError, match="feature shape"):
            acc.add(np.array([0]), np.ones((1, 3)))
        with pytest.raises(BankBuildError, match="does not match"):
            acc.add(np.array([0, 1]), np.ones((1, 2)))

    def test_non_finite_features_raise(self):
        acc = make_accumulator(n_stimuli=3, feature_shape=(1,))
        with pytest.raises(BankBuildError, match="non-finite"):
            acc.add(np.array([0]), np.array([[np.inf]]))

    def test_epsilon_clamps_zero_variance(self):
        acc = make_accumulator(n_stimuli=1, feature_shape=(2,))
        acc.add(np.array([0]), np.ones((1, 2)))
        acc.add(np.array([0]), np.ones((1, 2)))
        _, sigma, _ = acc.finalize(epsilon=1e-6)
        np.testing.assert_allclose(sigma[0], 1e-6, rtol=0, atol=1e-9)

    def test_missing_stimulus_zero_count(self):
        acc = make_accumulator(n_stimuli=4, feature_shape=(1,))
        acc.add(np.array([0, 2]), np.array([[1.0], [2.0]]))
        mu, sigma, count = acc.finalize(epsilon=1e-6)
        assert count[1] == 0 and count[3] == 0
        assert mu[1, 0] == 0.0 and sigma[1, 0] == 1e-6


class TestFeatureManifest:
    def test_build_feature_manifest(self):
        categories = ["Manipulated Images", "Natural Scenes", "Social Scenes", "Synthetic Images"]
        image = pd.DataFrame(
            {
                "stimulus_index": list(range(100)),
                "stimulus_id": [f"s{i:03d}.jpg" for i in range(100)],
                "category": [categories[i % 4] for i in range(100)],
            }
        )
        manifest = build_feature_manifest(image)
        assert list(manifest.columns) == ["stimulus_index", "stimulus_id", "category_id"]
        assert list(manifest.category_id[:8]) == [0, 1, 2, 3, 0, 1, 2, 3]

    def test_unknown_category_raises(self):
        image = pd.DataFrame(
            {
                "stimulus_index": [0],
                "stimulus_id": ["x.jpg"],
                "category": ["Unknown Category"],
            }
        )
        with pytest.raises(ValueError, match="unknown stimulus category"):
            build_feature_manifest(image)

    def test_manifest_requires_all_100_indices(self):
        image = pd.DataFrame(
            {
                "stimulus_index": [0, 1],
                "stimulus_id": ["a.jpg", "b.jpg"],
                "category": ["Natural Scenes", "Natural Scenes"],
            }
        )
        with pytest.raises(ValueError, match="exactly stimulus indices"):
            build_feature_manifest(image)


class TestPublication:
    def _arrays(self, include_token: bool = False) -> BankArrays:
        rng = np.random.default_rng(7)
        arrays = BankArrays(
            mu_trial=rng.normal(size=(100, 128)).astype(np.float32),
            sigma_trial=(1.0 + rng.random((100, 128))).astype(np.float32),
            count_trial=np.full(100, 5, dtype=np.int32),
        )
        if include_token:
            arrays.mu_token = rng.normal(size=(100, 192, 128)).astype(np.float32)
            arrays.sigma_token = (1.0 + rng.random((100, 192, 128))).astype(np.float32)
        return arrays

    def _feature_manifest(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stimulus_index": list(range(100)),
                "stimulus_id": [f"s{i:03d}.jpg" for i in range(100)],
                "category_id": [i % 4 for i in range(100)],
            }
        )

    def _provenance(self) -> dict:
        return {
            "stage1_checkpoint_path": "/tmp/checkpoint.pt",
            "stage1_checkpoint_sha256": "a" * 64,
            "stage1_run_id": "run",
            "stage1_config_hash": "b" * 64,
            "processed_dataset_metadata_sha256": "c" * 64,
            "dino_feature_manifest_sha256": "d" * 64,
            "cv_partition_sha256": {"train_trials": "e" * 64, "val_trials": "f" * 64},
            "source_checksums": {},
        }

    def _metadata(self) -> dict:
        return build_metadata(
            fold=0,
            crossfit_split=None,
            seed=2026,
            evaluation_regime="pilot_existing_stage1",
            estimator="mean",
            epsilon=1e-6,
            min_samples=2,
            include_fused_token_bank=False,
            include_heatmap_token_bank=False,
            provenance=self._provenance(),
            contributing_hc_subject_ids=[f"{i:03d}" for i in range(10)],
            forbidden_validation_subject_ids=[f"{i:03d}" for i in range(10, 20)],
            n_contributing_subjects=10,
            n_trials=500,
            array_shapes={},
            array_sha256={},
            stimulus_manifest_sha256="",
        )

    def test_publish_after_success_only(self, tmp_path: Path):
        bank_dir = tmp_path / "fold_0"
        publish_bank_dir(
            bank_dir,
            self._arrays(),
            feature_manifest=self._feature_manifest(),
            metadata=self._metadata(),
            expected_shapes=expected_array_shapes(False, False),
            min_samples=2,
            epsilon=1e-6,
        )
        for name in ("mu_trial.npy", "sigma_trial.npy", "count_trial.npy", "feature_manifest.csv", "metadata.json", "audit.json"):
            assert (bank_dir / name).is_file(), name
        audit = pd.read_json(bank_dir / "audit.json", typ="series")
        assert audit["checks"]["complete"] is True

    def test_failed_audit_leaves_no_directory(self, tmp_path: Path):
        bank_dir = tmp_path / "fold_0"
        arrays = self._arrays()
        arrays.sigma_trial = np.zeros_like(arrays.sigma_trial)  # below epsilon -> audit fails
        with pytest.raises(BankVerifyError):
            publish_bank_dir(
                bank_dir,
                arrays,
                feature_manifest=self._feature_manifest(),
                metadata=self._metadata(),
                expected_shapes=expected_array_shapes(False, False),
                min_samples=2,
                epsilon=1e-6,
            )
        assert not bank_dir.exists()
        assert not list(tmp_path.glob(".*.tmp"))

    def test_complete_bank_is_never_overwritten(self, tmp_path: Path):
        bank_dir = tmp_path / "fold_0"
        shapes = expected_array_shapes(False, False)
        publish_bank_dir(
            bank_dir, self._arrays(), feature_manifest=self._feature_manifest(),
            metadata=self._metadata(), expected_shapes=shapes, min_samples=2, epsilon=1e-6,
        )
        with pytest.raises(BankBuildError, match="already complete"):
            publish_bank_dir(
                bank_dir, self._arrays(), feature_manifest=self._feature_manifest(),
                metadata=self._metadata(), expected_shapes=shapes, min_samples=2, epsilon=1e-6,
            )

    def test_incomplete_bank_requires_flag_then_rebuilds(self, tmp_path: Path):
        bank_dir = tmp_path / "fold_0"
        shapes = expected_array_shapes(False, False)
        bank_dir.mkdir()
        (bank_dir / "mu_trial.npy").write_bytes(b"partial")
        with pytest.raises(BankBuildError, match="overwrite-incomplete"):
            publish_bank_dir(
                bank_dir, self._arrays(), feature_manifest=self._feature_manifest(),
                metadata=self._metadata(), expected_shapes=shapes, min_samples=2, epsilon=1e-6,
            )
        publish_bank_dir(
            bank_dir, self._arrays(), feature_manifest=self._feature_manifest(),
            metadata=self._metadata(), expected_shapes=shapes, min_samples=2, epsilon=1e-6,
            overwrite_incomplete=True,
        )
        assert (bank_dir / "audit.json").is_file()
        assert not (bank_dir / "mu_trial.npy").read_bytes() == b"partial"

    def test_publish_includes_token_arrays_when_enabled(self, tmp_path: Path):
        bank_dir = tmp_path / "fold_0"
        shapes = expected_array_shapes(True, False)
        publish_bank_dir(
            bank_dir, self._arrays(include_token=True), feature_manifest=self._feature_manifest(),
            metadata=self._metadata(), expected_shapes=shapes, min_samples=2, epsilon=1e-6,
        )
        assert (bank_dir / "mu_token.npy").is_file()
        assert (bank_dir / "sigma_token.npy").is_file()

    def test_bank_config_validation(self, tmp_path: Path):
        with pytest.raises(BankBuildError, match="estimator"):
            BankConfig.from_dict({"estimator": "median", "output_root": str(tmp_path)})
        with pytest.raises(BankBuildError, match="crossfit"):
            BankConfig.from_dict(
                {"crossfit_enabled": True, "crossfit_splits": 1, "output_root": str(tmp_path)}
            )
        with pytest.raises(BankBuildError, match="unknown"):
            BankConfig.from_dict({"output_root": str(tmp_path), "bogus": 1})
