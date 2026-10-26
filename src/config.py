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

# --- Chronological split (Milestone 6) ---------------------------------------
# Whole seasons, oldest -> newest. Every season in the feature table must appear in
# exactly one list. 2011-12 only builds history (form, Elo): its first matches are
# artificial cold starts. Validation and test are both 36-team-format seasons; the test
# season is scored once, at the end of Milestone 8.
WARMUP_SEASONS: list[str] = ["2011-12"]
TRAIN_SEASONS: list[str] = [f"{y}-{str(y + 1)[-2:]}" for y in range(2012, 2024)]  # 2012-13 to 2023-24
VALIDATION_SEASONS: list[str] = ["2024-25"]
TEST_SEASONS: list[str] = ["2025-26"]

# --- Data sources ------------------------------------------------------------
# Historical results: openfootball/champions-league (CC0-1.0 public domain).
OPENFOOTBALL_BASE_URL: str = (
    "https://raw.githubusercontent.com/openfootball/champions-league/master"
)
OPENFOOTBALL_RAW_DIR: Path = RAW_DATA_DIR / "openfootball"

# Seasons verified to exist in the repository (2011-12 to 2025-26).
HISTORICAL_SEASONS: list[str] = [f"{y}-{str(y + 1)[-2:]}" for y in range(2011, 2026)]

# Current-season fixtures: football-data.org API (free plan: current season only,
# 10 requests/minute). No published redistribution terms, so this folder is git-ignored.
FOOTBALL_DATA_BASE_URL: str = "https://api.football-data.org/v4"
FOOTBALL_DATA_RAW_DIR: Path = RAW_DATA_DIR / "football_data_org"
FOOTBALL_DATA_KEY_VAR: str = "FOOTBALL_DATA_API_KEY"  # name of the variable in .env
ENV_FILE: Path = PROJECT_ROOT / ".env"

# Second source for quality checks: English Wikipedia season articles (CC BY-SA 4.0).
# Pages are kept local (git-ignored) to keep licences separate; revision IDs are logged.
WIKIPEDIA_API_URL: str = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_RAW_DIR: Path = RAW_DATA_DIR / "wikipedia"
HTTP_USER_AGENT: str = (
    "champions-league-prediction-model/0.1 "
    "(educational project; https://github.com/EvansMaingi/champions-league-prediction-model)"
)

# Reference tables maintained by hand (reviewed, committed).
REFERENCE_DIR: Path = DATA_DIR / "reference"
