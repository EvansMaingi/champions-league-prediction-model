"""Download raw Champions League match data and save it unchanged to data/raw/.

Phase: Milestone 2 - Data collection.

Previous stage: none. This is the entry point of the pipeline.
Next stage: clean.py reads the files written here.

Rules: the data source and its licence must be verified before any code is written.
API keys are read from .env via python-dotenv, never hardcoded. Raw files are saved
exactly as received, with the download date recorded, and are never edited by hand.

TODO (planned functions):
- load_api_key(var_name: str) -> str
    Read a key from the environment (.env); raise a clear error if it is missing.
- fetch_matches(season: str) -> list[dict] | bytes
    Request one season of matches from the verified source.
- save_raw(content, filename: str) -> Path
    Write the response unchanged to RAW_DATA_DIR and return the file path.
- collect_seasons(seasons: list[str]) -> list[Path]
    Fetch and save several seasons; return the paths of the saved files.
"""
