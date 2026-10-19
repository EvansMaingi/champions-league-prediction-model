"""Central project configuration: filesystem paths and global settings.

Phase: used by every milestone.

Every other module imports its paths and constants from here instead of hardcoding
them, so the project runs from any machine or working directory, and a setting such
as the form window is changed in exactly one place.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
# This file lives at <root>/src/config.py, so the project root is two levels up.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"  # downloaded files, never edited by hand
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"  # cleaned data and feature tables
MODELS_DIR: Path = PROJECT_ROOT / "models"  # trained *.joblib files (git-ignored)
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# --- Global settings ---------------------------------------------------------
# Fixed seed so random parts (e.g. Random Forest bootstrapping) are reproducible.
RANDOM_SEED: int = 42

# Number of previous matches used for rolling "form" features.
FORM_WINDOW: int = 5
