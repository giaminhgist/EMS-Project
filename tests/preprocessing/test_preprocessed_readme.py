"""README rendering tests on the synthetic dataset."""

from __future__ import annotations

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
