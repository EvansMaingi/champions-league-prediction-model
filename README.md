# UEFA Champions League Match Prediction System

A machine learning project that estimates **home win / draw / away win** probabilities for
upcoming UEFA Champions League fixtures, trained on real historical match results.

It is built as a learning project: every stage of the ML pipeline, from data collection
to a web app, is implemented step by step and explained along the way.

> **Disclaimer:** Predictions are uncertain statistical estimates produced for educational
> and football-analysis purposes only. They are not guaranteed outcomes and are **not
> betting advice**.

## Project Specification

The full specification is in [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md). In short:

- **Goal:** predict the three-way result (`H` / `D` / `A`) of upcoming Champions League
  fixtures as probabilities, and explain the predictions and their limitations.
- **Pipeline:** problem definition → data collection → exploration → cleaning → feature
  engineering → chronological train/validation/test split → baseline → model training →
  evaluation → improvement → probability calibration → prediction → deployment → monitoring.
- **Models:** a `DummyClassifier` baseline, Logistic Regression, Random Forest and a
  gradient boosting classifier, all in scikit-learn Pipelines.
- **Evaluation:** accuracy, precision/recall, macro-F1, confusion matrix, multiclass log
  loss and Brier score, always compared with the baseline on the same held-out matches.
- **Key principles:** real, traceable data only; every feature must be known before
  kickoff; evaluation respects time order; preprocessing never learns from the test set;
  the training data cutoff and prediction timestamp are recorded; unknown teams and
  manually entered fixtures are clearly flagged.
- **App:** a Streamlit interface with a dashboard, fixture predictor, upcoming fixtures
  list and model insights.

## Tech Stack

| Purpose | Tools |
|---|---|
| Language | Python 3.12 |
| Data handling | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | scikit-learn |
| Model persistence | joblib |
| Web app | Streamlit |
| Data collection / config | requests, python-dotenv |
| Testing | pytest |
| Storage | CSV files |

## Project Structure

```
champions-league-prediction-model/
├── data/
│   ├── raw/                  original downloaded data, never edited by hand
│   └── processed/            cleaned match table and feature tables
├── docs/
│   └── PROJECT_SPEC.md       full project specification
├── notebooks/                exploratory data analysis
├── models/                   saved trained pipelines + metadata (*.joblib not in git)
├── src/
│   ├── config.py             project paths, random seed, form window size
│   ├── data/
│   │   ├── collect.py        download/import historical matches and fixtures
│   │   ├── clean.py          cleaning and team name standardisation
│   │   └── validate.py       data validation checks
│   ├── features/
│   │   └── build_features.py target creation + leakage-safe rolling pre-match features
│   ├── training/
│   │   ├── split.py          chronological train/validation/test split
│   │   ├── train.py          baseline + models with Pipelines/ColumnTransformer
│   │   └── evaluate.py       metrics, confusion matrix, log loss, Brier, calibration
│   └── prediction/
│       └── predict.py        predictions for upcoming fixtures
├── app/
│   └── streamlit_app.py      Streamlit web interface
├── tests/                    pytest test suite, one file per pipeline stage
├── .env.example              names of required environment variables (no real keys)
├── .gitignore
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Setup

Requires Python 3.11 or 3.12 and Git.

**1. Clone the repository**

```bash
git clone https://github.com/EvansMaingi/champions-league-prediction-model.git
cd champions-league-prediction-model
```

**2. Create and activate a virtual environment**

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Copy `.env.example` to `.env` and fill in any values it lists. `.env` is git-ignored.

**5. Check the installation**

```bash
python -c "import pandas, numpy, sklearn, streamlit, joblib; print('OK')"
pytest
```

## Data Sources and Licenses

| Data | Source | Licence / terms | In this repo? |
|---|---|---|---|
| Historical results, 2011-12 to 2025-26 (1,997 matches) | [openfootball/champions-league](https://github.com/openfootball/champions-league) | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain) | Yes: `data/raw/openfootball/` |
| Current-season fixtures and results | [football-data.org](https://www.football-data.org) API, free plan | No published redistribution terms | No: fetched locally into `data/raw/football_data_org/` (git-ignored) |

Thanks to the openfootball project for its public-domain data. Football data provided by
the Football-Data.org API.

**Collecting the data** (from the project root, with the virtual environment active):

```bash
python -m src.data.collect             # historical seasons (skips files already downloaded)
python -m src.data.collect fixtures    # one timestamped snapshot of current-season fixtures
```

The fixtures command needs a free football-data.org token in `.env` (see `.env.example`).
Raw files are saved unchanged, and each download is recorded in the folder's
`download_log.csv` (URL, size, SHA-256 hash, UTC time).

**Known limitations of the sources:**

- openfootball is maintained by volunteers; scores will be spot-checked against a second
  source during exploratory analysis.
- The football-data.org free plan serves only the current season (past seasons return
  HTTP 403), is limited to 10 requests per minute, and delays scores. Knockout fixtures
  appear only once they are scheduled.
- Team names differ between and within sources (e.g. "Bayern München" vs
  "FC Bayern München") and are standardised during cleaning.

## Training the Model

_To be completed (Milestones 6–8)._

## Running the App

_To be completed (Milestone 10)._

## Evaluation Results

_To be completed (Milestone 8). No results are reported until they come from running the code._

## Known Limitations

_To be completed._

## Future Improvements

_To be completed._

Planned experiments:

- **Older seasons vs era drift (Milestone 8):** test whether adding pre-2011 seasons
  (e.g. engsoccerdata's European Cup data, non-commercial licence) improves validation
  performance, or whether changes in the competition over time make old data harmful.
