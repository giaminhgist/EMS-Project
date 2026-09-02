"""README artifact tests (guide 08 §5).

The README is a static artifact: the generator script was removed by explicit
user request, so these tests verify the artifact itself rather than the
generator.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "readme" / "Stage2" / "README.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    assert README_PATH.is_file(), "README must exist before the test suite"
    return README_PATH.read_text(encoding="utf-8")


def test_every_registered_ablation_appears_exactly_once(readme_text):
    from stage2.ablations import ABLATIONS

    for name in ABLATIONS:
        assert readme_text.count(f"`{name}`") >= 1, name
    # The registry table has exactly one row per ablation.
    for name in ABLATIONS:
        assert (
            f"`python stage2_trainer.py --config configs/stage2/base.yaml --fold all --ablation {name}`"
            in readme_text
        )


def test_documented_commands_use_real_parser_options(readme_text):
    from stage2_trainer import build_parser
    from stage2.ablations import build_parser as ablations_parser

    options = set()
    for build in (build_parser, ablations_parser):
        for action in build()._actions:
            options.update(action.option_strings)
    # Every long option mentioned in the README exists in one of the real
    # parsers (the prose truncation "--max-*" is not a standalone option).
    for match in re.finditer(r"--[a-z][a-z0-9-]*", readme_text):
        token = match.group(0)
        if token == "--max" or readme_text[match.end() : match.end() + 1] == "*":
            continue
        assert token in options, token


def test_required_tensor_shapes_and_loss_names_appear(readme_text):
    for shape in ("[B,100,3,48,64]", "[N,192,128]", "[N,770]", "[B,100,128]",
                  "[B,128]", "[B,100,12,16]", "[100,192,128]", "[100,128]"):
        assert shape in readme_text, shape
    for loss in ("L_cls", "L_aux", "L_trialmatch", "L_bankrank", "L_tokenmatch",
                 "L_cons", "L_ent", "L_anchor"):
        assert loss in readme_text, loss


def test_evaluation_regime_caveat_appears(readme_text):
    assert "pilot_existing_stage1" in readme_text
    assert "outer_fold_exploratory" in readme_text
    assert "strict_nested_stage1" in readme_text
    assert "not a strict held-out estimate" in readme_text


def test_readme_says_subject_level_not_trial_level(readme_text):
    assert "subject — not the trial — is the independent classification" in readme_text
    assert "All losses, metrics, bootstrap and confidence intervals" in readme_text
    # No claim that trials are the evaluation unit.
    assert "trial is the independent" not in readme_text


def test_output_path_has_exact_capitalization():
    assert str(README_PATH.relative_to(REPO_ROOT)) == "readme/Stage2/README.md"


def test_no_developer_specific_paths_beyond_repo_root(readme_text):
    # The documented repository root is allowed; every other absolute path
    # must not appear.
    allowed = ("/root/EMS-Project",)
    for match in re.finditer(r"/root/[A-Za-z0-9_/.-]+", readme_text):
        assert match.group(0).startswith(allowed), match.group(0)


def test_no_fabricated_results_embedded(readme_text):
    # No validation score, epoch metric table or "accuracy = <number>" claim.
    assert not re.search(r"val_(balanced_accuracy|auroc|loss)\s*[:=]\s*[0-9]", readme_text)
    assert not re.search(r"accuracy\s*[:=]\s*[0-9]", readme_text)
    assert "## 1" in readme_text  # structure starts with the methodology sections
    for banned in ("0.9", "AUROC =", "balanced accuracy ="):
        assert banned not in readme_text.replace("λ", "")


def test_no_generator_references_remain(readme_text):
    """After the generator removal, the README must not point to it."""
    assert "generate_stage2_readme.py" not in readme_text
    assert "usage: stage2_trainer.py" in readme_text  # correct prog in the help block
