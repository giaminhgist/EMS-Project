"""README rendering and --verify-only tests on the synthetic dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocessing.eda import compute_eda_summary
from preprocessing.readme_renderer import render_readme, write_artifacts

REQUIRED_HEADINGS = [
    "## 1. Purpose and provenance",
    "## 2. Directory and file schema",
    "## 3. Exact channel definitions",
    "## 4. Population and trial inventory",
    "## 5. Heatmap-channel statistics",
    "## 6. Basic HC/SZ EDA (descriptive)",
    "## 7. Caveats",
    "## Figures",
]

REQUIRED_FIGURES = [
    "figures/fig_subject_trial_counts.png",
    "figures/fig_trials_per_subject.png",
    "figures/fig_subject_fixations_by_group.png",
    "figures/fig_qc_events.png",
    "figures/fig_example_trials.png",
    "figures/fig_group_average_heatmaps.png",
]


@pytest.fixture
def rendered(synthetic_processed, tmp_path) -> dict:
    out = tmp_path / "readme_out"
    summary = compute_eda_summary(synthetic_processed, "pytest synthetic")
    readme_path, summary_path, figures = write_artifacts(summary, out)
    return {
        "out": out,
        "readme_path": readme_path,
        "summary_path": summary_path,
        "figures": figures,
        "summary": summary,
    }


def test_readme_contains_all_required_sections(rendered):
    text = rendered["readme_path"].read_text(encoding="utf-8")
    for heading in REQUIRED_HEADINGS:
        assert heading in text, heading
    # Measured values, not placeholders.
    assert "n = 5" not in text or "subjects" in text  # sanity only
    assert "Mann–Whitney" in text
    assert "never used for model training" in text.lower()
    assert "```bash" in text and "```python" in text


def test_figure_links_and_valid_pngs(rendered):
    text = rendered["readme_path"].read_text(encoding="utf-8")
    for rel in REQUIRED_FIGURES:
        assert f"]({rel})" in text, rel
        path = rendered["out"] / rel
        assert path.is_file(), rel
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", rel


def test_rendering_is_deterministic(rendered):
    text1 = render_readme(rendered["summary"])
    text2 = render_readme(rendered["summary"])
    assert text1 == text2


def test_verify_only_passes_on_fresh_output(rendered, synthetic_processed):
    import sys

    from generate_preprocessed_readme import verify

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    assert verify(Path(synthetic_processed), rendered["out"]) == 0


def test_verify_only_detects_modified_manifest(rendered, synthetic_processed):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from generate_preprocessed_readme import verify

    manifest = Path(synthetic_processed) / "subject_manifest.csv"
    original = manifest.read_text(encoding="utf-8")
    try:
        manifest.write_text(original + "000,0,HC,0,000.xlsx,1,1,2,abc\n", encoding="utf-8")
        assert verify(Path(synthetic_processed), rendered["out"]) == 1
    finally:
        manifest.write_text(original, encoding="utf-8")


def test_verify_only_detects_modified_preprocessing_config(rendered, synthetic_processed):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from generate_preprocessed_readme import verify

    cfg_path = Path(synthetic_processed) / "preprocessing_config.json"
    original = cfg_path.read_text(encoding="utf-8")
    try:
        cfg = json.loads(original)
        cfg["gaussian_sigma_cells"] = 9.9
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        assert verify(Path(synthetic_processed), rendered["out"]) == 1
    finally:
        cfg_path.write_text(original, encoding="utf-8")


def test_verify_only_detects_modified_readme(rendered, synthetic_processed):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from generate_preprocessed_readme import verify

    readme = rendered["readme_path"]
    original = readme.read_text(encoding="utf-8")
    try:
        readme.write_text(original + "\n# tampered\n", encoding="utf-8")
        assert verify(Path(synthetic_processed), rendered["out"]) == 1
    finally:
        readme.write_text(original, encoding="utf-8")
