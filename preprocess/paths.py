"""Dataset path helpers for the CROSS reproducibility package.

Set EEG_DATA_ROOT to the directory that contains the dataset folders, e.g.

    EEG_DATA_ROOT=/path/to/eeg_datasets

The loaders then read paths such as:

    /path/to/eeg_datasets/Zhou2016
    /path/to/eeg_datasets/BCIC_IV_2a

If EEG_DATA_ROOT is not set, the project-local ./data directory is used.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("EEG_DATA_ROOT", PROJECT_ROOT / "data")).expanduser()


def dataset_path(*parts: str) -> str:
    """Return an absolute path under EEG_DATA_ROOT."""
    return str((DATA_ROOT.joinpath(*parts)).resolve())
