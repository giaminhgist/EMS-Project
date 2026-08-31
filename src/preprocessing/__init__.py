"""EMS preprocessing: fixation workbooks -> manifests and three-channel heatmaps."""

from .config import ConfigError, PreprocessingConfig
from .heatmaps import HeatmapParams, build_trial_heatmap
from .pipeline import PipelineOptions, PipelineReport, run_pipeline
from .storage import TrialRecord, TrialStore

__all__ = [
    "ConfigError",
    "PreprocessingConfig",
    "HeatmapParams",
    "build_trial_heatmap",
    "PipelineOptions",
    "PipelineReport",
    "run_pipeline",
    "TrialRecord",
    "TrialStore",
]
