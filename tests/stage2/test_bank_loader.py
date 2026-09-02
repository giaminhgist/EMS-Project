"""Normative bank loader tests (guide 04 §10, §12)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import (
    make_bank_fixture,
    make_cv_metadata,
    make_cv_root,
    make_processed_fixture,
    make_stage2_config,
)
from stage2.bank import FULL_BANK_ID, NormativeBankStore, audit_data_boundary
from stage2.bank_builder import BankVerifyError

HC_IDS = [f"{i:03d}" for i in range(8)]
SZ_TRAIN = ["100", "101"]
VAL_IDS = ["102", "103"]
SUBJECTS = {sid: 0 for sid in HC_IDS} | {sid: 1 for sid in SZ_TRAIN + VAL_IDS}
TRAIN_IDS = HC_IDS + SZ_TRAIN
OBSERVED = {sid: list(range(6)) for sid in SUBJECTS}


def make_fixture(tmp_path, include_token: bool = True):
    processed = make_processed_fixture(tmp_path, SUBJECTS, OBSERVED, with_dino=True)
    cv_root = make_cv_root(tmp_path, SUBJECTS, TRAIN_IDS, VAL_IDS)
    make_cv_metadata(cv_root, processed)
    dino_root = tmp_path / "stimulus_features" / "dino_vits16"
    bank_root, registry = make_bank_fixture(
        tmp_path,
        processed_root=processed,
        cv_root=cv_root,
        hc_ids=HC_IDS,
        sz_ids=SZ_TRAIN,
        forbidden_ids=VAL_IDS,
        crossfit_splits=2,
        include_token=include_token,
        dino_root=dino_root,
    )
    cfg = make_stage2_config(
        processed_root=processed,
        cv_root=cv_root,
        bank_root=bank_root,
        registry_path=registry,
    )
    return cfg


class TestStoreInit:
    def test_loads_and_exposes_banks(self, tmp_path):
        cfg = make_fixture(tmp_path)
        store = NormativeBankStore(cfg, 0)
        assert store.train_mode == "crossfit"
        assert store.has_token_banks is True
        assert store.contributors_full == set(HC_IDS)
        assert store.forbidden == set(VAL_IDS)
        assert set(store.excluded_by_split) == {0, 1}
        assert all(len(c) == 4 for c in store.contributors_by_split.values())

    def test_checksum_mismatch_failure(self, tmp_path):
        cfg = make_fixture(tmp_path)
        # Tamper a processed input after the bank recorded its checksum.
        path = cfg.paths.processed_root / "subject_manifest.csv"
        path.write_text(path.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
        with pytest.raises(BankVerifyError, match="checksum mismatch"):
            NormativeBankStore(cfg, 0)

    def test_checkpoint_sha_mismatch_failure(self, tmp_path):
        cfg = make_fixture(tmp_path)
        import yaml

        from stage2.bank_builder import load_checkpoint_registry

        registry = load_checkpoint_registry(cfg.bank.checkpoint_registry)
        registry["folds"]["0"]["sha256"] = "f" * 64
        cfg.bank.checkpoint_registry.write_text(yaml.safe_dump(registry), encoding="utf-8")
        with pytest.raises(BankVerifyError, match="registry"):
            NormativeBankStore(cfg, 0)

    def test_missing_token_arrays_only_when_required(self, tmp_path):
        cfg = make_fixture(tmp_path, include_token=False)
        store = NormativeBankStore(cfg, 0)
        assert store.has_token_banks is False
        with pytest.raises(BankVerifyError, match="token banks required"):
            NormativeBankStore(cfg, 0, require_token_banks=True)

    def test_self_inclusion_failure(self, tmp_path):
        cfg = make_fixture(tmp_path)
        meta_path = cfg.bank.root / "fold_0" / "crossfit" / "split_0" / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["contributing_hc_subject_ids"] = sorted(
            set(meta["contributing_hc_subject_ids"]) | {meta["crossfit_excluded_hc_subject_ids"][0]}
        )
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(BankVerifyError, match="self-inclusion"):
            NormativeBankStore(cfg, 0)

    def test_manifest_order_mismatch_failure(self, tmp_path):
        cfg = make_fixture(tmp_path)
        manifest_path = cfg.bank.root / "fold_0" / "feature_manifest.csv"
        manifest = pd.read_csv(manifest_path)
        manifest = manifest.sample(frac=1.0, random_state=0).reset_index(drop=True)
        manifest.to_csv(manifest_path, index=False)
        with pytest.raises(BankVerifyError, match="stimulus order"):
            NormativeBankStore(cfg, 0)

    def test_assignment_completeness_failure(self, tmp_path):
        cfg = make_fixture(tmp_path)
        assignment_path = cfg.bank.root / "fold_0" / "crossfit" / "subject_assignment.csv"
        assignment = pd.read_csv(assignment_path, dtype=str)
        assignment = assignment[assignment.subject_id != HC_IDS[0]]
        assignment.to_csv(assignment_path, index=False)
        with pytest.raises(BankVerifyError, match="assignment"):
            NormativeBankStore(cfg, 0)


class TestBankSelection:
    def test_bank_for_subject_rules(self, tmp_path):
        cfg = make_fixture(tmp_path)
        store = NormativeBankStore(cfg, 0)
        hc_by_split = store.assignment.set_index("subject_id").bank_split_id.to_dict()
        for sid in HC_IDS:
            assert store.bank_for_subject(sid, "train", int(hc_by_split[sid])) == int(hc_by_split[sid])
        assert store.bank_for_subject("102", "val", None) == FULL_BANK_ID
        assert store.bank_split_id_for("102", "val") is None
        assert store.bank_split_id_for(HC_IDS[0], "train") == int(hc_by_split[HC_IDS[0]])

    def test_full_mode_train_uses_full_bank(self, tmp_path):
        cfg = make_fixture(tmp_path)
        full_cfg = make_stage2_config(
            processed_root=cfg.paths.processed_root,
            cv_root=cfg.paths.cv_root,
            bank_root=cfg.bank.root,
            registry_path=cfg.bank.checkpoint_registry,
            train_mode="full_self_included",
        )
        store = NormativeBankStore(full_cfg, 0)
        assert store.bank_for_subject(HC_IDS[0], "train", None) == FULL_BANK_ID
        assert store.bank_split_id_for(HC_IDS[0], "train") is None


class TestGather:
    def test_exact_stimulus_gather_full_bank(self, tmp_path):
        cfg = make_fixture(tmp_path)
        store = NormativeBankStore(cfg, 0)
        mu_full = np.load(cfg.bank.root / "fold_0" / "mu_trial.npy")
        si = np.array([0, 5, 99, 3])
        slots = np.array([0, 0, 1, 2])
        gathered = store.gather_trials(
            stimulus_indices=torch.tensor(si),
            subject_slots=torch.tensor(slots),
            subject_bank_ids=None,
            split="val",
        )
        np.testing.assert_allclose(gathered.mu_trial.numpy(), mu_full[si], rtol=0, atol=0)
        assert gathered.bank_ids.tolist() == [FULL_BANK_ID] * 4
        assert not gathered.mu_trial.requires_grad

    def test_mixed_crossfit_ids_within_one_batch(self, tmp_path):
        cfg = make_fixture(tmp_path)
        store = NormativeBankStore(cfg, 0)
        assignment = store.assignment.set_index("subject_id").bank_split_id.to_dict()
        # Subject 0 -> split a, subject 1 -> split b (8 HC, 2 splits).
        split_a = int(assignment[HC_IDS[0]])
        split_b = 1 - split_a
        mu_split_a = np.load(cfg.bank.root / "fold_0" / "crossfit" / f"split_{split_a}" / "mu_trial.npy")
        mu_split_b = np.load(cfg.bank.root / "fold_0" / "crossfit" / f"split_{split_b}" / "mu_trial.npy")
        si = np.array([2, 9, 4, 8])
        slots = np.array([0, 0, 1, 1])
        # Per-SUBJECT bank ids: subject 0 uses split_a, subject 1 uses split_b.
        subject_bank_ids = np.array([split_a, split_b])
        gathered = store.gather_trials(
            stimulus_indices=torch.tensor(si),
            subject_slots=torch.tensor(slots),
            subject_bank_ids=torch.tensor(subject_bank_ids),
            split="train",
        )
        expected = np.vstack([mu_split_a[si[:2]], mu_split_b[si[2:]]])
        np.testing.assert_allclose(gathered.mu_trial.numpy(), expected, rtol=0, atol=0)
        assert gathered.bank_ids.tolist() == [split_a, split_a, split_b, split_b]
        assert gathered.mu_token is not None and gathered.mu_token.shape == (4, 192, 128)

    def test_validation_always_full_bank(self, tmp_path):
        cfg = make_fixture(tmp_path)
        store = NormativeBankStore(cfg, 0)
        mu_full = np.load(cfg.bank.root / "fold_0" / "mu_trial.npy")
        si = np.array([1, 2])
        gathered = store.gather_trials(
            stimulus_indices=torch.tensor(si),
            subject_slots=torch.tensor([0, 1]),
            subject_bank_ids=torch.tensor([0, 1]),  # ignored for val
            split="val",
        )
        np.testing.assert_allclose(gathered.mu_trial.numpy(), mu_full[si], rtol=0, atol=0)
        assert gathered.bank_ids.tolist() == [FULL_BANK_ID, FULL_BANK_ID]

    def test_unknown_bank_id_failure(self, tmp_path):
        cfg = make_fixture(tmp_path)
        store = NormativeBankStore(cfg, 0)
        with pytest.raises(BankVerifyError, match="unknown bank ids"):
            store.gather_trials(
                stimulus_indices=torch.tensor([0]),
                subject_slots=torch.tensor([0]),
                subject_bank_ids=torch.tensor([7]),
                split="train",
            )

    def test_out_of_range_stimulus_failure(self, tmp_path):
        cfg = make_fixture(tmp_path)
        store = NormativeBankStore(cfg, 0)
        with pytest.raises(BankVerifyError, match="out of bank manifest range"):
            store.gather_trials(
                stimulus_indices=torch.tensor([100]),
                subject_slots=torch.tensor([0]),
                subject_bank_ids=None,
                split="val",
            )


class FakeDataset:
    def __init__(self, ids: list[str], labels: list[int]):
        self.subject_ids = ids
        self._labels = labels

    def subject_labels(self) -> list[int]:
        return self._labels


class TestLeakageAudit:
    def test_audit_ok(self, tmp_path):
        cfg = make_fixture(tmp_path)
        store = NormativeBankStore(cfg, 0)
        train = FakeDataset(TRAIN_IDS, [SUBJECTS[s] for s in TRAIN_IDS])
        val = FakeDataset(VAL_IDS, [SUBJECTS[s] for s in VAL_IDS])
        audit = audit_data_boundary(train, val, store)
        assert audit["status"] == "ok"
        assert audit["train_val_disjoint"] and audit["no_val_subject_in_bank"]
        assert audit["crossfit_self_exclusion"] and audit["bank_contributor_counts_comparable"]

    def test_audit_detects_overlap(self, tmp_path):
        cfg = make_fixture(tmp_path)
        store = NormativeBankStore(cfg, 0)
        train = FakeDataset(TRAIN_IDS, [SUBJECTS[s] for s in TRAIN_IDS])
        val = FakeDataset(TRAIN_IDS[:1], [0])
        audit = audit_data_boundary(train, val, store)
        assert audit["train_val_disjoint"] is False
        assert audit["status"] == "fail"
