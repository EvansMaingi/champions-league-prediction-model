"""Predict home win / draw / away win probabilities for upcoming fixtures.

Phase: Milestone 9 - Upcoming fixture predictions.

Previous stage: train.py / evaluate.py saved the selected model to models/.
Next stage: app/streamlit_app.py calls these functions to show predictions.

Features for a fixture are built from matches played BEFORE its kickoff, using the
same logic as training. Unknown teams (no history) are flagged and not silently
predicted. Manually entered fixtures are labelled "unverified". Every prediction
records its timestamp and the model's training data cutoff.

TODO (planned functions):
- load_model(path: Path) -> tuple[Pipeline, dict]
    Load the joblib model and its metadata (training data cutoff, feature list).
- build_fixture_features(fixtures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame
    Pre-match features for each fixture from past matches only.
- find_unknown_teams(fixtures: pd.DataFrame, history: pd.DataFrame) -> set[str]
    Teams that have no match history.
- predict_fixtures(fixtures: pd.DataFrame, model: Pipeline, metadata: dict) -> pd.DataFrame
    p_home, p_draw, p_away per fixture, plus flags (unknown_team, unverified),
    prediction timestamp and training data cutoff.
"""
