"""Streamlit web app: show match outcome probabilities for upcoming fixtures.

Phase: Milestone 10 - Streamlit web app.

Previous stage: src/prediction/predict.py produces probabilities from the saved model.
Next stage: none. This is what the end user sees.

The app displays a disclaimer (educational estimates, not betting advice), the
training data cutoff, the prediction timestamp, and clear flags for unknown teams
and unverified (manually entered) fixtures. It contains no modelling logic of its
own; it only calls functions from src/.

Run (from Milestone 10): streamlit run app/streamlit_app.py

TODO (planned functions):
- load_resources() -> tuple[Pipeline, dict, pd.DataFrame]
    Load the model, metadata and match history (cached with st.cache_resource).
- render_disclaimer() -> None
    Show the educational-use disclaimer and data cutoff.
- render_fixture_predictions(predictions: pd.DataFrame) -> None
    Table / bars of p_home, p_draw, p_away with any flags.
- render_manual_fixture_form() -> pd.DataFrame | None
    Let the user enter a fixture; mark it as unverified.
- main() -> None
    Lay out the page and wire the pieces together.
"""
