from __future__ import annotations
from pathlib import Path
import os

def _find_root_from_markers(start: Path) -> Path:
    markers = {"pyproject.toml", "requirements.txt", ".git"}
    for p in [start, *start.parents]:
        if any((p / m).exists() for m in markers):
            return p
    return start

def get_project_root() -> Path:
    env_root = os.environ.get("RUNNER_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    here = Path(__file__).resolve()
    found = _find_root_from_markers(here)
    if found != here:
        return found.resolve()

    return here.parents[2].resolve()

PROJECT_ROOT = get_project_root()
DATASET_ENVIRONMENT_LSTM = PROJECT_ROOT/"datasets"/"environment_lstm"

BASE_MODEL = PROJECT_ROOT/"trainings"/"environment_lstm"/"base_model"
BASE_MODEL_METRICS = PROJECT_ROOT/"metrics"/"environment_lstm"/"base_model"

PRUNED_MODEL = PROJECT_ROOT/"trainings"/"environment_lstm"/"pruned_model"
PRUNED_MODEL_METRICS = PROJECT_ROOT/"metrics"/"environment_lstm"/"pruned_model"

QUANTIZED_MODEL = PROJECT_ROOT/"trainings"/"environment_lstm"/"quantized_model"
QUANTIZED_MODEL_METRICS = PROJECT_ROOT/"metrics"/"environment_lstm"/"quantized_model"

HEADER_GENERATOR = PROJECT_ROOT/"trainings"/"environment_lstm"/"header_generator"

SCALERS_EXPORT = PROJECT_ROOT/"trainings"/"environment_lstm"/"scalers_exporter"

def p(*parts: str | Path) -> Path:
    return PROJECT_ROOT.joinpath(*map(str, parts))
