"""Identifier-contract tests: leading zeros, non-contiguous IDs, trial UIDs."""

from __future__ import annotations

import pandas as pd
import pytest

from preprocessing.identifiers import (
    SubjectIdentity,
    resolve_group_label,
    trial_uid,
)
from preprocessing.inventory import inventory_stimuli


def test_leading_zero_subject_ids_survive_csv_roundtrip(tmp_path):
    ids = ["000", "005", "013", "102", "203"]
    df = pd.DataFrame({"subject_id": ids})
    csv_path = tmp_path / "subjects.csv"
    df.to_csv(csv_path, index=False)
    back = pd.read_csv(csv_path, dtype={"subject_id": str})
    assert list(back.subject_id) == ids
    assert back.subject_id.iloc[0] == "000"  # no int coercion


def test_leading_zero_subject_ids_survive_parquet_roundtrip(tmp_path):
    ids = ["000", "005", "013", "102", "203"]
    df = pd.DataFrame({"subject_id": ids, "label": [0, 0, 0, 0, 1]})
    pq_path = tmp_path / "subjects.parquet"
    df.to_parquet(pq_path, index=False)
    back = pd.read_parquet(pq_path)
    assert list(back.subject_id) == ids
    assert back.subject_id.iloc[0] == "000"


def test_noncontiguous_ids_do_not_index_arrays(tmp_path):
    # Stimulus indices must come from the manifest order (category, basename),
    # never from the numeric part of the image name.
    from tests.preprocessing.conftest import write_image

    img_root = tmp_path / "Images"
    for name in ["art_999.jpg", "soc_007.jpg", "cat_012.jpg"]:
        cat = {"art_999.jpg": "C1", "soc_007.jpg": "C1", "cat_012.jpg": "C2"}[name]
        write_image(img_root / cat / name)
    stimuli = inventory_stimuli(img_root)
    assert [s.stimulus_index for s in stimuli] == [0, 1, 2]
    assert [s.stimulus_id for s in stimuli] == ["art_999.jpg", "soc_007.jpg", "cat_012.jpg"]
    assert stimuli[0].stimulus_index != 999


def test_subject_identity_preserves_leading_zeros_and_derives_labels():
    ident = SubjectIdentity.from_stem("000")
    assert ident.subject_id == "000"
    assert ident.subject_numeric_id == 0
    assert (ident.group, ident.label) == ("HC", 0)

    ident_sz = SubjectIdentity.from_stem("203")
    assert ident_sz.subject_id == "203"
    assert (ident_sz.group, ident_sz.label) == ("SZ", 1)


def test_label_rule_split_boundary():
    assert resolve_group_label(199) == ("HC", 0)
    assert resolve_group_label(200) == ("SZ", 1)
    assert resolve_group_label(303) == ("SZ", 1)


def test_trial_uid_is_deterministic_sha256_prefix():
    uid1 = trial_uid("000", "a1.jpg")
    uid2 = trial_uid("000", "a1.jpg")
    assert uid1 == uid2
    assert len(uid1) == 20
    assert all(c in "0123456789abcdef" for c in uid1)
    assert trial_uid("000", "a1.jpg") != trial_uid("000", "a2.jpg")
    assert trial_uid("000", "a1.jpg") != trial_uid("005", "a1.jpg")
    # Case and separator sensitivity.
    assert trial_uid("000", "A1.jpg") != trial_uid("000", "a1.jpg")
    assert trial_uid("00", "0\0a1.jpg") != trial_uid("000", "a1.jpg")
